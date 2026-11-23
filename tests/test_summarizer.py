"""Task 3: AI summarization — schema, word limit, grounding checks, caching. No network calls."""

from __future__ import annotations

import copy
import json
import sqlite3
from types import SimpleNamespace

from backend.config import AI_DISCLAIMER, SUMMARY_WORD_LIMIT
from backend.store import bundle_content_hash
from backend.summarize.cache import SummaryCache
from backend.summarize.summarizer import (
    PROMPT_VERSION, ClaudeBackend, ClinicalSummary, RulesBackend, build_prompt, summarize_bundle,
)


def _summary(**kw) -> ClinicalSummary:
    base = dict(chief_concern="Shortness of breath.", key_diagnoses=["Acute on chronic systolic heart failure"],
                recent_media=[], flagged_anomalies=["Sodium 131 mmol/L (low)"], narrative="Admitted for heart failure.",
                confidence="high", confidence_rationale="Complete records.")
    base.update(kw)
    return ClinicalSummary(**base)


class FakeBackend:
    model = "fake-model"

    def __init__(self, *responses: ClinicalSummary):
        self.responses = list(responses)
        self.calls: list[str | None] = []

    def summarize(self, prompt, feedback=None):
        self.calls.append(feedback)
        return self.responses[min(len(self.calls), len(self.responses)) - 1]


def test_prompt_contains_all_records_and_data_quality_notes(bundles):
    prompt = build_prompt(bundles["pat-001002"], ["given: name conflict"])
    assert prompt.count("<record ") == 5  # discharge summary, telephone note, CXR, 2 lab panels
    assert "Hyponatremia" in prompt and "<data_quality_notes>" in prompt


def test_rule_based_summary_has_required_fields_and_disclaimer(bundles):
    s = summarize_bundle("pat-001005", bundles["pat-001005"], RulesBackend())
    assert s.chief_concern != "Not documented"
    assert "Left breast mass, BI-RADS 4 on imaging" in s.key_diagnoses
    assert any("BI-RADS 4C" in a for a in s.flagged_anomalies)
    assert s.disclaimer == AI_DISCLAIMER and "not a clinical decision" in s.disclaimer
    assert s.word_count <= SUMMARY_WORD_LIMIT
    assert s.record_hash == bundle_content_hash(bundles["pat-001005"])


def test_over_limit_summary_is_retried_with_feedback(bundles):
    long = _summary(narrative="word " * 300)
    backend = FakeBackend(long, _summary())
    s = summarize_bundle("pat-001002", bundles["pat-001002"], backend)
    assert len(backend.calls) == 2 and "under" in backend.calls[1]
    assert s.word_count <= SUMMARY_WORD_LIMIT and not s.quality_warnings


def test_over_limit_twice_is_trimmed_and_flagged(bundles):
    long = _summary(narrative=" ".join(f"Sentence number {i} about the patient." for i in range(60)))
    s = summarize_bundle("pat-001002", bundles["pat-001002"], FakeBackend(long, long))
    assert s.word_count <= SUMMARY_WORD_LIMIT
    assert any("trimmed" in w for w in s.quality_warnings)


def test_grounding_check_flags_invented_facts(bundles):
    invented = _summary(key_diagnoses=["Acute on chronic systolic heart failure", "Metastatic pancreatic carcinoma"],
                        flagged_anomalies=["Troponin 4.7 ng/mL"],
                        recent_media=[{"record_id": "NOT-A-RECORD", "kind": "imaging", "date": "2025-01-01",
                                       "description": "x"}])
    s = summarize_bundle("pat-001002", bundles["pat-001002"], FakeBackend(invented))
    joined = " ".join(s.quality_warnings)
    assert "pancreatic" in joined and "4.7" in joined and "NOT-A-RECORD" in joined
    assert "systolic heart failure" not in joined


def test_cache_hit_skips_backend_and_content_change_misses(bundles):
    conn = sqlite3.connect(":memory:")
    cache = SummaryCache(conn)
    backend = FakeBackend(_summary())
    bundle = bundles["pat-001002"]

    def run(b):
        h = bundle_content_hash(b)
        if (hit := cache.get("pat-001002", h, backend.model, PROMPT_VERSION)) is not None:
            return hit
        s = summarize_bundle("pat-001002", b, backend)
        cache.put(s)
        return s

    run(bundle)
    run(copy.deepcopy(bundle))          # identical content -> cache hit
    assert len(backend.calls) == 1
    changed = copy.deepcopy(bundle)
    changed["entry"][0]["resource"]["gender"] = "unknown"
    run(changed)                        # different content -> new call
    assert len(backend.calls) == 2
    assert cache.latest("pat-001002").model == "fake-model"


def test_claude_backend_parses_structured_output():
    payload = _summary().model_dump()
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(stop_reason="end_turn", stop_details=None,
                               content=[SimpleNamespace(type="text", text=json.dumps(payload))])

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    out = ClaudeBackend(client=client).summarize("prompt")
    assert out == _summary()
    assert captured["output_config"]["format"]["type"] == "json_schema"
    assert "Use only facts stated in the records" in captured["system"]
