"""AI clinical summaries of each patient's FHIR Bundle.

Flow per patient
  bundle -> context (records as tagged text) -> cache lookup (patient_id + bundle content hash)
         -> LLM (structured JSON output) -> word-limit + grounding checks -> cache -> StoredSummary

If no LLM API credentials are available, a deterministic rule-based extractor produces a summary in the
same schema (clearly labelled, confidence "low") so the rest of the platform still runs end to end.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from backend.config import AI_DISCLAIMER, LLM_MODEL, SUMMARY_WORD_LIMIT
from backend.store import bundle_content_hash

log = logging.getLogger(__name__)

PROMPT_VERSION = "v4"  # bump to invalidate cached summaries when the prompt changes
RULES_MODEL = "rule-based-fallback"


# --------------------------------------------------------------------------------------
# Output schema
# --------------------------------------------------------------------------------------
class MediaItem(BaseModel):
    record_id: str
    kind: Literal["imaging", "lab", "note"]
    date: str
    description: str


class ClinicalSummary(BaseModel):
    """What the model returns."""

    chief_concern: str
    key_diagnoses: list[str]
    recent_media: list[MediaItem]
    flagged_anomalies: list[str]
    narrative: str
    confidence: Literal["high", "medium", "low"]
    confidence_rationale: str


class StoredSummary(ClinicalSummary):
    """What we store and serve: model output + provenance, disclaimer and quality checks."""

    patient_id: str
    record_hash: str
    model: str
    prompt_version: str = PROMPT_VERSION
    generated_at: str
    word_count: int
    disclaimer: str = AI_DISCLAIMER
    quality_warnings: list[str] = Field(default_factory=list)


# Hand-written JSON schema for structured outputs (additionalProperties: false everywhere, all fields required).
SUMMARY_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["chief_concern", "key_diagnoses", "recent_media", "flagged_anomalies", "narrative",
                 "confidence", "confidence_rationale"],
    "properties": {
        "chief_concern": {"type": "string"},
        "key_diagnoses": {"type": "array", "items": {"type": "string"}},
        "recent_media": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["record_id", "kind", "date", "description"],
            "properties": {
                "record_id": {"type": "string"},
                "kind": {"type": "string", "enum": ["imaging", "lab", "note"]},
                "date": {"type": "string"},
                "description": {"type": "string"},
            }}},
        "flagged_anomalies": {"type": "array", "items": {"type": "string"}},
        "narrative": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "confidence_rationale": {"type": "string"},
    },
}

SYSTEM_PROMPT = f"""You summarize a single patient's electronic health records for clinicians who are searching \
a records system. Your summary helps them decide which source documents to open; it is never used as a \
clinical decision.

Accuracy rules:
- Use only facts stated in the records provided. Do not infer diagnoses, causes, severity or treatments \
that are not written down. Copy lab values, units and reference ranges exactly.
- If something is not documented, say "Not documented" rather than guessing.
- Keep the clinical wording of the source (e.g. keep "BI-RADS 4C", "EF 30%").

Fields:
- chief_concern: the presenting problem of the most recent encounter, one sentence.
- key_diagnoses: diagnoses explicitly listed in assessments or report impressions, most important first, \
no more than 5, each a short phrase.
- recent_media: the most recent imaging reports, lab panels and notes (no more than 4), newest first, using \
the record id, kind and date exactly as given in the record tags, with a description of 12 words or fewer.
- flagged_anomalies: abnormal lab values (value, unit, reference range), suspicious or critical imaging \
findings, follow-up the records explicitly recommend, and any data-quality notes provided. Empty list if none.
- narrative: 2-3 sentences tying the above together.
- confidence: "high" if the records are complete and consistent, "medium" if some information is missing \
or dated, "low" if the records are sparse or contradictory. Explain in confidence_rationale (one sentence).

Length: everything you write, across all fields combined, must stay under {SUMMARY_WORD_LIMIT - 30} words."""


# --------------------------------------------------------------------------------------
# Bundle -> prompt context
# --------------------------------------------------------------------------------------
def _decode(att: dict[str, Any]) -> str:
    try:
        return base64.b64decode(att.get("data", "")).decode("utf-8")
    except Exception:  # noqa: BLE001
        return ""


def bundle_records(bundle: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """(patient resource, [{id, kind, date, title, text}]) newest first."""
    patient: dict[str, Any] = {}
    records: list[dict[str, str]] = []
    for e in bundle.get("entry", []):
        r = e["resource"]
        rt = r["resourceType"]
        if rt == "Patient":
            patient = r
        elif rt == "DocumentReference":
            att = r["content"][0]["attachment"]
            dtype, desc = r["type"].get("text", "Note"), r.get("description", "")
            records.append({"id": r["id"], "kind": "note", "date": (r.get("date") or "")[:10],
                            "title": desc if desc == dtype or not desc else f"{dtype}: {desc}",
                            "text": _decode(att)})
        elif rt == "DiagnosticReport":
            is_lab = r["category"][0]["coding"][0]["code"] == "LAB"
            text = _decode(r["presentedForm"][0]) if r.get("presentedForm") else r.get("conclusion", "")
            records.append({"id": r["id"], "kind": "lab" if is_lab else "imaging",
                            "date": r.get("effectiveDateTime", "")[:10], "title": r["code"].get("text", ""),
                            "text": text})
    records.sort(key=lambda x: (x["date"], x["id"]), reverse=True)
    return patient, records


def build_prompt(bundle: dict[str, Any], data_quality_notes: list[str] | None = None) -> str:
    patient, records = bundle_records(bundle)
    name = (patient.get("name") or [{}])[0].get("text", "Unknown")
    parts = [f"<patient name=\"{name}\" birth_date=\"{patient.get('birthDate', 'unknown')}\" "
             f"gender=\"{patient.get('gender', 'unknown')}\" />"]
    for r in records:
        parts.append(f"<record id=\"{r['id']}\" kind=\"{r['kind']}\" date=\"{r['date'] or 'unknown'}\" "
                     f"title=\"{r['title']}\">\n{r['text']}\n</record>")
    if data_quality_notes:
        parts.append("<data_quality_notes>\n" + "\n".join(f"- {n}" for n in data_quality_notes) +
                     "\n</data_quality_notes>")
    parts.append("Summarize this patient's records following the rules and fields described.")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------------------
# Quality checks
# --------------------------------------------------------------------------------------
def render_text(s: ClinicalSummary) -> str:
    media = "; ".join(f"{m.kind} {m.date}: {m.description}" for m in s.recent_media)
    return (f"Chief concern: {s.chief_concern}\nKey diagnoses: {'; '.join(s.key_diagnoses)}\n"
            f"Recent media: {media}\nFlagged anomalies: {'; '.join(s.flagged_anomalies) or 'None'}\n"
            f"{s.narrative}")


def word_count(s: ClinicalSummary) -> int:
    return len(render_text(s).split())


_STOP = {"with", "without", "from", "this", "that", "and", "the", "for", "left", "right", "acute", "chronic",
         "type", "stage", "moderate", "mild", "status", "post"}


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 3 and w not in _STOP}


def grounding_warnings(s: ClinicalSummary, records: list[dict[str, str]],
                       extra_source: list[str] | None = None) -> list[str]:
    """Cheap hallucination guard: diagnoses and anomaly numbers must be traceable to the source text."""
    source = " ".join([r["text"] + " " + r["title"] for r in records] + (extra_source or [])).lower()
    source_words = _content_words(source)
    ids = {r["id"] for r in records}
    warnings: list[str] = []
    for dx in s.key_diagnoses:
        words = _content_words(dx)
        if words and len(words & source_words) / len(words) < 0.6:
            warnings.append(f"Diagnosis not clearly found in source records: '{dx}'")
    for m in s.recent_media:
        if m.record_id not in ids:
            warnings.append(f"recent_media cites unknown record id '{m.record_id}'")
    for a in s.flagged_anomalies:
        for num in re.findall(r"\d+(?:\.\d+)?", a):
            if num not in source:
                warnings.append(f"Value {num} in anomaly not found in source: '{a}'")
                break
    return warnings


def _trim_narrative(s: ClinicalSummary, limit: int) -> ClinicalSummary:
    """Last resort if the model overshoots twice: drop trailing narrative sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", s.narrative)
    while sentences and word_count(s.model_copy(update={"narrative": " ".join(sentences)})) > limit:
        sentences.pop()
    return s.model_copy(update={"narrative": " ".join(sentences)})


# --------------------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------------------
class SummaryBackend(Protocol):
    model: str

    def summarize(self, prompt: str, feedback: str | None = None) -> ClinicalSummary: ...


class LLMBackend:
    """LLM summarizer. Implemented with the Anthropic Messages API (model set by EHR_LLM_MODEL)."""

    def __init__(self, model: str = LLM_MODEL, client: Any = None):
        import anthropic

        self.model = model
        self.client = client or anthropic.Anthropic()

    def summarize(self, prompt: str, feedback: str | None = None) -> ClinicalSummary:
        content = prompt if feedback is None else f"{prompt}\n\n{feedback}"
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SUMMARY_JSON_SCHEMA}},
            # On a safety-classifier refusal, let the API retry on a fallback model chosen per refusal category.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(f"Model declined to summarize (stop_details={response.stop_details})")
        if response.stop_reason == "max_tokens":
            raise RuntimeError("Summary truncated at max_tokens")
        text = next(b.text for b in response.content if b.type == "text")
        return ClinicalSummary.model_validate(json.loads(text))


_CC = re.compile(r"Chief Complaint:\s*(.+)")
_ASSESS = re.compile(r"Assessment:\s*\n((?:\s*\d+\.\s*.+\n?)+)")
_SUSPICIOUS = re.compile(r"(BI-RADS 4|BI-RADS 5|malignan|nodule|infarct|follow-up|recommended|suspicious|"
                         r"hemorrhage|embolism|fracture|consolidation|edema)", re.IGNORECASE)


class RulesBackend:
    """Deterministic extractive summarizer used when no LLM credentials are configured (and in tests)."""

    model = RULES_MODEL

    def summarize(self, prompt: str, feedback: str | None = None) -> ClinicalSummary:
        records = [
            {"id": m.group(1), "kind": m.group(2), "date": m.group(3), "title": m.group(4), "text": m.group(5)}
            for m in re.finditer(r'<record id="([^"]+)" kind="([^"]+)" date="([^"]+)" title="([^"]*)">\n(.*?)\n</record>',
                                 prompt, re.DOTALL)
        ]
        dq = re.search(r"<data_quality_notes>\n(.*?)\n</data_quality_notes>", prompt, re.DOTALL)
        notes = [r for r in records if r["kind"] == "note"]
        cc = next((m.group(1).strip() for r in notes if (m := _CC.search(r["text"]))), "Not documented")
        dx: list[str] = []
        for r in notes:
            if m := _ASSESS.search(r["text"] + "\n"):
                for line in m.group(1).strip().splitlines():
                    item = re.sub(r"^\s*\d+\.\s*", "", line).strip()
                    if item and item not in dx:
                        dx.append(item)
        anomalies: list[str] = []
        for r in records:
            if r["kind"] == "lab":
                anomalies += [f"{ln.lstrip('- ').strip()}" for ln in r["text"].splitlines()
                              if "[HIGH]" in ln or "[LOW]" in ln or "[ABNORMAL]" in ln]
            elif r["kind"] == "imaging" and (m := re.search(r"Impression:\s*(.+)", r["text"])):
                if _SUSPICIOUS.search(m.group(1)):
                    anomalies.append(f"{r['title']}: {m.group(1).strip()}")
        if dq:
            anomalies += [ln.lstrip("- ").strip() for ln in dq.group(1).splitlines()]
        media = [MediaItem(record_id=r["id"], kind=r["kind"], date=r["date"], description=r["title"][:80])
                 for r in records[:4]]
        narrative = (f"Records describe {'; '.join(dx[:2]) or 'no listed diagnoses'}. "
                     f"{len(records)} records on file; {len(anomalies)} flagged item(s).")
        s = ClinicalSummary(chief_concern=cc, key_diagnoses=dx[:5], recent_media=media,
                            flagged_anomalies=anomalies[:6], narrative=narrative, confidence="low",
                            confidence_rationale="Rule-based extraction without an LLM; verify against records.")
        return _trim_narrative(s, SUMMARY_WORD_LIMIT) if word_count(s) > SUMMARY_WORD_LIMIT else s


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def summarize_bundle(patient_id: str, bundle: dict[str, Any], backend: SummaryBackend,
                     data_quality_notes: list[str] | None = None) -> StoredSummary:
    prompt = build_prompt(bundle, data_quality_notes)
    _, records = bundle_records(bundle)
    summary = backend.summarize(prompt)
    warnings: list[str] = []
    n = word_count(summary)
    if n > SUMMARY_WORD_LIMIT:  # one corrective retry, then trim deterministically
        summary = backend.summarize(prompt, feedback=f"Your previous summary was {n} words. Rewrite it so all "
                                                     f"fields combined total under {SUMMARY_WORD_LIMIT - 30} words.")
        if word_count(summary) > SUMMARY_WORD_LIMIT:
            summary = _trim_narrative(summary, SUMMARY_WORD_LIMIT)
            warnings.append("Summary exceeded the word limit twice; narrative was trimmed.")
    warnings += grounding_warnings(summary, records, data_quality_notes)
    return StoredSummary(
        **summary.model_dump(), patient_id=patient_id, record_hash=bundle_content_hash(bundle),
        model=backend.model, generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        word_count=word_count(summary), quality_warnings=warnings,
    )
