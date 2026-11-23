"""AI clinical summaries of each patient's FHIR Bundle.

Flow per patient
  bundle -> SummaryInput (records extracted once; JSON prompt) -> cache lookup
         -> Claude (structured JSON output) -> word-limit + grounding checks -> cache -> StoredSummary

If no Anthropic credentials are available, a deterministic rule-based extractor produces a summary in the
same schema (clearly labelled, confidence "low") so the rest of the platform still runs end to end.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from backend.config import AI_DISCLAIMER, CLAUDE_MODEL, SUMMARY_WORD_LIMIT
from backend.store import bundle_content_hash, decode_attachment

log = logging.getLogger(__name__)

PROMPT_VERSION = "v6"  # bump to invalidate cached summaries when the prompt or summary logic changes
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
    record_hash: str                 # hash of the Bundle's clinical content
    notes_hash: str                  # hash of the data-quality notes that were part of the input
    model: str                       # model that actually produced the summary (may be a fallback model)
    requested_model: str             # model the pipeline asked for; part of the cache key
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

The input is JSON: the patient, the records (newest first, each with id, kind, date, title and text) and \
data-quality notes produced while cleaning the source data.

Fields:
- chief_concern: the presenting problem of the most recent encounter, one sentence.
- key_diagnoses: diagnoses explicitly listed in assessments or report impressions, most important first, \
no more than 5, each a short phrase.
- recent_media: the most recent imaging reports, lab panels and notes (no more than 4), newest first, using \
the record id, kind and date exactly as given in the input, with a description of 12 words or fewer.
- flagged_anomalies: abnormal lab values (value, unit, reference range), suspicious or critical imaging \
findings, follow-up the records explicitly recommend, and every data-quality note provided. Empty list if none.
- narrative: 2-3 sentences tying the above together.
- confidence: "high" if the records are complete and consistent, "medium" if some information is missing \
or dated, "low" if the records are sparse or contradictory. Explain in confidence_rationale (one sentence).

Length: everything you write, across all fields combined, must stay under {SUMMARY_WORD_LIMIT - 30} words."""


# --------------------------------------------------------------------------------------
# Bundle -> summary input
# --------------------------------------------------------------------------------------
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
                            "text": decode_attachment(att)})
        elif rt == "DiagnosticReport":
            is_lab = r["category"][0]["coding"][0]["code"] == "LAB"
            text = decode_attachment(r["presentedForm"][0]) if r.get("presentedForm") else r.get("conclusion", "")
            records.append({"id": r["id"], "kind": "lab" if is_lab else "imaging",
                            "date": r.get("effectiveDateTime", "")[:10], "title": r["code"].get("text", ""),
                            "text": text})
    records.sort(key=lambda x: (x["date"], x["id"]), reverse=True)
    return patient, records


@dataclass
class SummaryInput:
    """Everything a backend needs, extracted from the Bundle once. Backends read `records` directly; only the
    LLM uses `prompt`, which is JSON, so titles or text containing quotes or angle brackets can't break it."""

    patient_id: str
    patient: dict[str, Any]
    records: list[dict[str, str]]
    data_quality_notes: list[str] = field(default_factory=list)
    record_hash: str = ""
    notes_hash: str = ""

    @property
    def prompt(self) -> str:
        name = (self.patient.get("name") or [{}])[0].get("text", "Unknown")
        payload = {
            "patient": {"name": name, "birth_date": self.patient.get("birthDate", "unknown"),
                        "gender": self.patient.get("gender", "unknown")},
            "records": [{**r, "date": r["date"] or "unknown"} for r in self.records],
            "data_quality_notes": self.data_quality_notes,
        }
        return (json.dumps(payload, ensure_ascii=False, indent=1) +
                "\n\nSummarize this patient's records following the rules and fields described.")


def notes_hash(notes: list[str] | None) -> str:
    return hashlib.sha256(json.dumps(sorted(notes or [])).encode()).hexdigest()


def prepare_input(patient_id: str, bundle: dict[str, Any], data_quality_notes: list[str] | None = None) -> SummaryInput:
    patient, records = bundle_records(bundle)
    notes = list(data_quality_notes or [])
    return SummaryInput(patient_id=patient_id, patient=patient, records=records, data_quality_notes=notes,
                        record_hash=bundle_content_hash(bundle), notes_hash=notes_hash(notes))


# --------------------------------------------------------------------------------------
# Quality checks
# --------------------------------------------------------------------------------------
def render_text(s: ClinicalSummary) -> str:
    """Every field the model writes, as one text. The word limit is measured on this."""
    media = "; ".join(f"{m.kind} {m.date}: {m.description}" for m in s.recent_media)
    return (f"Chief concern: {s.chief_concern}\nKey diagnoses: {'; '.join(s.key_diagnoses)}\n"
            f"Recent media: {media}\nFlagged anomalies: {'; '.join(s.flagged_anomalies) or 'None'}\n"
            f"{s.narrative}\nConfidence: {s.confidence}. {s.confidence_rationale}")


def word_count(s: ClinicalSummary) -> int:
    return len(render_text(s).split())


_STOP = {"with", "without", "from", "this", "that", "and", "the", "for", "left", "right", "acute", "chronic",
         "type", "stage", "moderate", "mild", "status", "post"}
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_LIST_NUMBER = re.compile(r"(?m)^\s*\d+[.)]\s")  # "1. Hypertension" list markers are not clinical values


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 3 and w not in _STOP}


def _numbers(text: str) -> set[float]:
    """Whole numeric tokens as values, so '4.7' does not match inside '14.7' and '131' equals '131.0'."""
    return {float(n) for n in _NUMBER.findall(_LIST_NUMBER.sub(" ", text))}


def grounding_warnings(s: ClinicalSummary, records: list[dict[str, str]],
                       extra_source: list[str] | None = None) -> list[str]:
    """Cheap hallucination guard: diagnoses and anomaly numbers must be traceable to the source text."""
    texts = [r["text"] + "\n" + r["title"] for r in records] + (extra_source or [])
    source_words = _content_words(" ".join(texts))
    source_numbers = set().union(*(_numbers(t) for t in texts)) if texts else set()
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
        missing = [n for n in _NUMBER.findall(a) if float(n) not in source_numbers]
        if missing:
            warnings.append(f"Value {missing[0]} in anomaly not found in source: '{a}'")
    return warnings


def _first_words(text: str, n: int) -> str:
    words = text.split()
    return text if len(words) <= n else " ".join(words[:n]) + "…"


def enforce_word_limit(s: ClinicalSummary, limit: int, protected: list[str] | None = None) -> tuple[ClinicalSummary, list[str]]:
    """Deterministic last resort when the model overshoots twice. Shortens in order of least clinical value
    until the whole summary fits: narrative sentences, older media, diagnoses beyond the top 3, long items,
    then the remaining unprotected anomalies. Data-quality notes (`protected`) are kept. Returns the
    summary and the names of the fields that were cut."""
    protected_set = set(protected or [])
    cut: list[str] = []

    def over(x: ClinicalSummary) -> bool:
        return word_count(x) > limit

    sentences = re.split(r"(?<=[.!?])\s+", s.narrative)
    while over(s) and sentences and sentences != [""]:
        sentences.pop()
        s = s.model_copy(update={"narrative": " ".join(sentences)})
        cut.append("narrative")
    while over(s) and len(s.recent_media) > 1:
        s = s.model_copy(update={"recent_media": s.recent_media[:-1]})
        cut.append("recent_media")
    while over(s) and len(s.key_diagnoses) > 3:
        s = s.model_copy(update={"key_diagnoses": s.key_diagnoses[:-1]})
        cut.append("key_diagnoses")
    if over(s):
        s = s.model_copy(update={
            "chief_concern": _first_words(s.chief_concern, 25),
            "confidence_rationale": _first_words(s.confidence_rationale, 20),
            "key_diagnoses": [_first_words(d, 8) for d in s.key_diagnoses],
            "recent_media": [m.model_copy(update={"description": _first_words(m.description, 8)})
                             for m in s.recent_media],
            "flagged_anomalies": [a if a in protected_set else _first_words(a, 15) for a in s.flagged_anomalies],
        })
        cut.append("long items shortened")
    while over(s) and any(a not in protected_set for a in s.flagged_anomalies):
        idx = max(i for i, a in enumerate(s.flagged_anomalies) if a not in protected_set)
        s = s.model_copy(update={"flagged_anomalies": s.flagged_anomalies[:idx] + s.flagged_anomalies[idx + 1:]})
        cut.append("flagged_anomalies")
    while over(s) and (s.recent_media or len(s.key_diagnoses) > 1):
        if s.recent_media:
            s = s.model_copy(update={"recent_media": s.recent_media[:-1]})
        else:
            s = s.model_copy(update={"key_diagnoses": s.key_diagnoses[:-1]})
        cut.append("lists")
    return s, list(dict.fromkeys(cut))


# --------------------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------------------
class SummaryBackend(Protocol):
    model: str

    def summarize(self, inp: SummaryInput, feedback: str | None = None) -> tuple[ClinicalSummary, str]:
        """Returns the summary and the name of the model that actually produced it."""
        ...


class ClaudeBackend:
    def __init__(self, model: str = CLAUDE_MODEL, client: Any = None):
        import anthropic

        self.model = model
        self.client = client or anthropic.Anthropic()

    def summarize(self, inp: SummaryInput, feedback: str | None = None) -> tuple[ClinicalSummary, str]:
        content = inp.prompt if feedback is None else f"{inp.prompt}\n\n{feedback}"
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
        block = next((b for b in response.content if b.type == "text"), None)
        if block is None:
            raise RuntimeError(f"Response had no text block (stop_reason={response.stop_reason}, "
                               f"blocks={[b.type for b in response.content]})")
        # With server-side fallbacks the answering model can differ from the one requested.
        served_by = getattr(response, "model", None) or self.model
        return ClinicalSummary.model_validate(json.loads(block.text)), served_by


_CC = re.compile(r"Chief Complaint:\s*(.+)")
_ASSESS = re.compile(r"Assessment:\s*\n((?:\s*\d+\.\s*.+\n?)+)")
_FLAGGED = re.compile(r"\[(CRITICAL )?(HIGH|LOW|ABNORMAL)\]")
_SUSPICIOUS = re.compile(r"(BI-RADS 4|BI-RADS 5|malignan|nodule|infarct|follow-up|recommended|suspicious|"
                         r"hemorrhage|embolism|fracture|consolidation|edema)", re.IGNORECASE)
MAX_CLINICAL_ANOMALIES = 6


class RulesBackend:
    """Deterministic extractive summarizer used when no LLM credentials are configured (and in tests)."""

    model = RULES_MODEL

    def summarize(self, inp: SummaryInput, feedback: str | None = None) -> tuple[ClinicalSummary, str]:
        records = inp.records
        notes = [r for r in records if r["kind"] == "note"]
        cc = next((m.group(1).strip() for r in notes if (m := _CC.search(r["text"]))), "Not documented")
        dx: list[str] = []
        for r in notes:
            if m := _ASSESS.search(r["text"] + "\n"):
                for line in m.group(1).strip().splitlines():
                    item = re.sub(r"^\s*\d+\.\s*", "", line).strip()
                    if item and item not in dx:
                        dx.append(item)
        clinical: list[str] = []
        for r in records:
            if r["kind"] == "lab":
                clinical += [ln.lstrip("- ").strip() for ln in r["text"].splitlines() if _FLAGGED.search(ln)]
            elif r["kind"] == "imaging" and (m := re.search(r"Impression:\s*(.+)", r["text"])):
                if _SUSPICIOUS.search(m.group(1)):
                    clinical.append(f"{r['title']}: {m.group(1).strip()}")
        # Clinical anomalies are capped; data-quality notes are always kept in full.
        anomalies = clinical[:MAX_CLINICAL_ANOMALIES] + inp.data_quality_notes
        media = [MediaItem(record_id=r["id"], kind=r["kind"], date=r["date"], description=r["title"][:80])
                 for r in records[:4]]
        narrative = (f"Records describe {'; '.join(dx[:2]) or 'no listed diagnoses'}. "
                     f"{len(records)} records on file; {len(clinical) + len(inp.data_quality_notes)} flagged item(s).")
        s = ClinicalSummary(chief_concern=cc, key_diagnoses=dx[:5], recent_media=media,
                            flagged_anomalies=anomalies, narrative=narrative, confidence="low",
                            confidence_rationale="Rule-based extraction without an LLM; verify against records.")
        return s, self.model


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def summarize_bundle(patient_id: str, bundle: dict[str, Any], backend: SummaryBackend,
                     data_quality_notes: list[str] | None = None) -> StoredSummary:
    inp = prepare_input(patient_id, bundle, data_quality_notes)
    summary, served_by = backend.summarize(inp)
    warnings: list[str] = []
    n = word_count(summary)
    if n > SUMMARY_WORD_LIMIT:  # one corrective retry with the previous output, then trim deterministically
        feedback = (f"Your previous summary, below, was {n} words. Shorten it so that all fields combined "
                    f"(including confidence_rationale) total under {SUMMARY_WORD_LIMIT - 30} words. Keep every "
                    f"data-quality note and the most clinically important findings.\n\nPrevious summary:\n"
                    f"{summary.model_dump_json()}")
        summary, served_by = backend.summarize(inp, feedback=feedback)
        if word_count(summary) > SUMMARY_WORD_LIMIT:
            summary, cut = enforce_word_limit(summary, SUMMARY_WORD_LIMIT, protected=inp.data_quality_notes)
            warnings.append(f"Summary exceeded the word limit twice; shortened deterministically "
                            f"({', '.join(cut)}).")
    warnings += grounding_warnings(summary, inp.records, inp.data_quality_notes)
    return StoredSummary(
        **summary.model_dump(), patient_id=patient_id, record_hash=inp.record_hash, notes_hash=inp.notes_hash,
        model=served_by, requested_model=backend.model,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        word_count=word_count(summary), quality_warnings=warnings,
    )
