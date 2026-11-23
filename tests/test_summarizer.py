"""Task 3: AI summarization — schema, word limit, grounding checks, caching. No network calls."""

from __future__ import annotations

import copy
import json
import sqlite3
from types import SimpleNamespace

import pytest

from backend.config import AI_DISCLAIMER, SUMMARY_WORD_LIMIT
from backend.store import bundle_content_hash
from backend.summarize import summarizer as SM
from backend.summarize.cache import SummaryCache
from backend.summarize.summarizer import (
    PROMPT_VERSION, RULES_MODEL, ClinicalSummary, LLMBackend, RulesBackend, SummaryInput, StoredSummary,
    enforce_word_limit, grounding_warnings, notes_hash, prepare_input, summarize_bundle, word_count,
)


def _summary(**kw) -> ClinicalSummary:
    base = dict(chief_concern="Shortness of breath.", key_diagnoses=["Acute on chronic systolic heart failure"],
                recent_media=[], flagged_anomalies=["Sodium 131 mmol/L (low)"], narrative="Admitted for heart failure.",
                confidence="high", confidence_rationale="Complete records.")
    base.update(kw)
    return ClinicalSummary(**base)


class FakeBackend:
    model = "fake-model"

    def __init__(self, *responses: ClinicalSummary, served_by: str | None = None):
        self.responses = list(responses)
        self.calls: list[str | None] = []
        self.served_by = served_by or self.model

    def summarize(self, inp, feedback=None):
        self.calls.append(feedback)
        return self.responses[min(len(self.calls), len(self.responses)) - 1], self.served_by


def _client(content, model="claude-opus-5-5", captured=None):
    def create(**kwargs):
        if captured is not None:
            captured.update(kwargs)
        return SimpleNamespace(stop_reason="end_turn", stop_details=None, model=model, content=content)
    return SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))


def _lab_input(lines: list[str], notes: list[str] | None = None, title: str = "Chem") -> SummaryInput:
    text = "Lab panel: Chem\n" + "\n".join(lines)
    return SummaryInput(patient_id="p", patient={}, records=[{"id": "LAB-1", "kind": "lab", "date": "2025-01-01",
                                                              "title": title, "text": text}],
                        data_quality_notes=notes or [])


# ---------------------------------------------------------------- prompt and schema
def test_prompt_is_json_with_all_records_and_data_quality_notes(bundles):
    inp = prepare_input("pat-001002", bundles["pat-001002"], ["given: name conflict"])
    payload = json.loads(inp.prompt.rsplit("\n\nSummarize", 1)[0])
    assert len(payload["records"]) == 5  # discharge summary, telephone note, CXR, 2 lab panels
    assert any("Hyponatremia" in r["text"] for r in payload["records"])
    assert payload["data_quality_notes"] == ["given: name conflict"]


def test_rule_based_summary_has_required_fields_and_disclaimer(bundles):
    s = summarize_bundle("pat-001005", bundles["pat-001005"], RulesBackend())
    assert s.chief_concern != "Not documented"
    assert "Left breast mass, BI-RADS 4 on imaging" in s.key_diagnoses
    assert any("BI-RADS 4C" in a for a in s.flagged_anomalies)
    assert s.disclaimer == AI_DISCLAIMER and "not a clinical decision" in s.disclaimer
    assert s.word_count <= SUMMARY_WORD_LIMIT
    assert s.record_hash == bundle_content_hash(bundles["pat-001005"])


def test_rule_based_summary_flags_critical_lab_values():
    s, _ = RulesBackend().summarize(_lab_input(["- Potassium: 7.1 mmol/L (ref 3.5-5.1) [CRITICAL HIGH]"]))
    assert any("Potassium" in a for a in s.flagged_anomalies)


def test_quotes_and_brackets_in_titles_do_not_drop_records(bundles):  # review finding 4
    b = copy.deepcopy(bundles["pat-001002"])
    lab = next(e["resource"] for e in b["entry"] if e["resource"]["resourceType"] == "DiagnosticReport"
               and "Cardiac" in e["resource"]["code"]["text"])
    lab["code"]["text"] = 'Cardiac "STAT" Panel <draft>'
    s = summarize_bundle("pat-001002", b, RulesBackend())
    assert any("natriuretic" in a for a in s.flagged_anomalies)
    payload = json.loads(prepare_input("pat-001002", b).prompt.rsplit("\n\nSummarize", 1)[0])
    assert any(r["title"] == 'Cardiac "STAT" Panel <draft>' for r in payload["records"])


def test_data_quality_notes_are_never_cut(bundles):  # review finding 6
    inp = _lab_input([f"- Test{i}: {i} u (ref 0) [HIGH]" for i in range(9)], notes=["given: conflicting names"])
    s, _ = RulesBackend().summarize(inp)
    assert "given: conflicting names" in s.flagged_anomalies
    assert len([a for a in s.flagged_anomalies if a.startswith("Test")]) == SM.MAX_CLINICAL_ANOMALIES


def test_bundle_records_runs_once_per_patient(bundles, monkeypatch):  # review finding 10
    calls = []
    original = SM.bundle_records
    monkeypatch.setattr(SM, "bundle_records", lambda b: calls.append(1) or original(b))
    summarize_bundle("pat-001002", bundles["pat-001002"], RulesBackend())
    assert len(calls) == 1


# ---------------------------------------------------------------- word limit
def test_word_count_includes_confidence_rationale():  # review finding 9
    assert word_count(_summary(confidence_rationale="extra " * 50)) == word_count(_summary()) - 2 + 50


def test_over_limit_summary_is_retried_with_previous_output(bundles):  # review finding 9
    long = _summary(narrative="word " * 300)
    backend = FakeBackend(long, _summary())
    s = summarize_bundle("pat-001002", bundles["pat-001002"], backend)
    assert len(backend.calls) == 2
    assert "under" in backend.calls[1] and long.model_dump_json() in backend.calls[1]
    assert s.word_count <= SUMMARY_WORD_LIMIT and not s.quality_warnings


def test_over_limit_twice_is_trimmed_and_flagged(bundles):
    long = _summary(narrative=" ".join(f"Sentence number {i} about the patient." for i in range(60)))
    s = summarize_bundle("pat-001002", bundles["pat-001002"], FakeBackend(long, long))
    assert s.word_count <= SUMMARY_WORD_LIMIT
    assert any("shortened" in w and "narrative" in w for w in s.quality_warnings)


def test_limit_enforced_even_when_lists_cause_the_overrun(bundles):  # review finding 3
    long = _summary(key_diagnoses=[f"diagnosis number {i} with several extra descriptive words" for i in range(40)],
                    flagged_anomalies=[f"anomaly {i} with quite a few extra words attached" for i in range(30)]
                    + ["MRN MRN-001070 merged into MRN-001007"])
    s = summarize_bundle("pat-001002", bundles["pat-001002"], FakeBackend(long, long),
                         data_quality_notes=["MRN MRN-001070 merged into MRN-001007"])
    assert s.word_count <= SUMMARY_WORD_LIMIT
    assert "MRN MRN-001070 merged into MRN-001007" in s.flagged_anomalies


def test_enforce_word_limit_keeps_protected_notes():
    s = _summary(flagged_anomalies=["keep me"] + [f"drop {i} " + "w " * 20 for i in range(20)])
    out, cut = enforce_word_limit(s, 60, protected=["keep me"])
    assert word_count(out) <= 60 and "keep me" in out.flagged_anomalies and "flagged_anomalies" in cut


# ---------------------------------------------------------------- grounding
def test_grounding_check_flags_invented_facts(bundles):
    invented = _summary(key_diagnoses=["Acute on chronic systolic heart failure", "Metastatic pancreatic carcinoma"],
                        flagged_anomalies=["Troponin 4.7 ng/mL"],
                        recent_media=[{"record_id": "NOT-A-RECORD", "kind": "imaging", "date": "2025-01-01",
                                       "description": "x"}])
    s = summarize_bundle("pat-001002", bundles["pat-001002"], FakeBackend(invented))
    joined = " ".join(s.quality_warnings)
    assert "pancreatic" in joined and "4.7" in joined and "NOT-A-RECORD" in joined
    assert "systolic heart failure" not in joined


@pytest.mark.parametrize("anomaly,flagged", [("Troponin 4.7 ng/mL", True), ("WBC 14.7", False),
                                             ("WBC 14.70", False), ("Sodium 1 mmol/L", True)])
def test_number_grounding_matches_whole_values(anomaly, flagged):  # review finding 5
    records = [{"id": "X", "kind": "lab", "date": "", "title": "CBC", "text": "Assessment:\n1. Anemia\nWBC 14.7"}]
    assert bool(grounding_warnings(_summary(key_diagnoses=[], flagged_anomalies=[anomaly]), records)) is flagged


# ---------------------------------------------------------------- caching
def test_cache_hit_skips_backend_and_content_or_notes_change_misses(bundles):  # review finding 2
    cache = SummaryCache(sqlite3.connect(":memory:"))
    backend = FakeBackend(_summary())
    bundle = bundles["pat-001002"]

    def run(b, notes=None):
        hit = cache.get("pat-001002", bundle_content_hash(b), notes_hash(notes), backend.model, PROMPT_VERSION)
        if hit is not None:
            return hit
        s = summarize_bundle("pat-001002", b, backend, notes)
        cache.put(s)
        return s

    run(bundle)
    run(copy.deepcopy(bundle))                       # identical inputs -> cache hit
    assert len(backend.calls) == 1
    run(bundle, ["given: name conflict"])            # new data-quality note -> new call
    assert len(backend.calls) == 2
    changed = copy.deepcopy(bundle)
    changed["entry"][0]["resource"]["gender"] = "unknown"
    run(changed)                                     # different content -> new call
    assert len(backend.calls) == 3


def _stored(model: str, prompt_version: str, record_hash: str) -> StoredSummary:
    return StoredSummary(**_summary().model_dump(), patient_id="p", record_hash=record_hash, notes_hash="n",
                         model=model, requested_model=model, prompt_version=prompt_version, generated_at="t",
                         word_count=1)


def test_latest_honours_prompt_version_and_newest_bundle():  # review finding 1
    cache = SummaryCache(sqlite3.connect(":memory:"))
    cache.put(_stored("llm-model", "old-version", "h1"))
    cache.put(_stored(RULES_MODEL, PROMPT_VERSION, "h1"))
    assert cache.latest("p", "h1").model == RULES_MODEL          # stale-prompt LLM summary is not served
    cache.put(_stored("llm-model", PROMPT_VERSION, "h1"))
    assert cache.latest("p", "h1").model == "llm-model"          # same inputs: LLM preferred over rules

    cache2 = SummaryCache(sqlite3.connect(":memory:"))
    cache2.put(_stored("llm-model", PROMPT_VERSION, "OLD"))
    cache2.put(_stored(RULES_MODEL, PROMPT_VERSION, "NEW"))
    assert cache2.latest("p").record_hash == "NEW"               # no hash given: newest bundle content wins


def test_old_cache_schema_is_rebuilt():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE summaries (patient_id TEXT, record_hash TEXT, model TEXT, prompt_version TEXT, "
                 "summary_json TEXT, created_at TEXT)")
    SummaryCache(conn).put(_stored("m", PROMPT_VERSION, "h"))
    assert SummaryCache(conn).latest("p", "h") is not None


# ---------------------------------------------------------------- LLM backend (mocked client)
def test_llm_backend_parses_structured_output(bundles):
    captured: dict = {}
    text = SimpleNamespace(type="text", text=json.dumps(_summary().model_dump()))
    out, served_by = LLMBackend(client=_client([text], captured=captured)).summarize(
        prepare_input("pat-001002", bundles["pat-001002"]))
    assert out == _summary() and served_by == "claude-opus-5-5"
    assert captured["output_config"]["format"]["type"] == "json_schema"
    assert "Use only facts stated in the records" in captured["system"]


def test_stored_model_is_the_model_that_answered(bundles):  # review finding 7
    text = SimpleNamespace(type="text", text=json.dumps(_summary().model_dump()))
    backend = LLMBackend(model="claude-opus-5-5", client=_client([text], model="claude-opus-5"))
    s = summarize_bundle("pat-001002", bundles["pat-001002"], backend)
    assert s.model == "claude-opus-5" and s.requested_model == "claude-opus-5-5"


def test_response_without_text_block_raises_clear_error(bundles):  # review finding 8
    backend = LLMBackend(client=_client([SimpleNamespace(type="thinking", thinking="")]))
    with pytest.raises(RuntimeError, match="no text block"):
        backend.summarize(prepare_input("pat-001002", bundles["pat-001002"]))
