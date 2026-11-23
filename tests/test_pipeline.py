"""scripts/run_pipeline.py: backend selection and how summary failures are handled (code review of PR #8)."""

from __future__ import annotations

import json
import logging
import sys
from types import SimpleNamespace

import anthropic
import httpx
import pytest

import run_pipeline
from backend import config
from backend.summarize.summarizer import RULES_MODEL, RulesBackend


def _auth_error() -> anthropic.AuthenticationError:
    response = httpx.Response(401, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
    return anthropic.AuthenticationError("invalid x-api-key", response=response, body=None)


def make_llm(fail_check: Exception | None = None, fail_for: dict[str, Exception] | None = None,
             fail_from_call: tuple[int, Exception] | None = None):
    """A stand-in for the LLM backend: rule-based content labelled as an LLM, with scripted failures."""
    calls = {"n": 0}

    def retrieve(model):
        if fail_check:
            raise fail_check
        return SimpleNamespace(id=model)

    class FakeLLM(RulesBackend):
        model = "fake-llm"

        def __init__(self):
            self.client = SimpleNamespace(models=SimpleNamespace(retrieve=retrieve))

        def summarize(self, inp, feedback=None):
            calls["n"] += 1
            if fail_from_call and calls["n"] >= fail_from_call[0]:
                raise fail_from_call[1]
            if fail_for and inp.patient_id in fail_for:
                raise fail_for[inp.patient_id]
            s, _ = super().summarize(inp, feedback)
            return s, self.model

    return FakeLLM


@pytest.fixture
def pipeline(raw_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "ehr.db")
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "out")

    def run(*args, llm=None):
        if llm is not None:
            monkeypatch.setattr(run_pipeline, "LLMBackend", llm)
        monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--raw-dir", str(raw_dir), "--skip-index", *args])
        run_pipeline.main()
        return {pid: s["model"] for pid, s in json.loads((tmp_path / "out" / "summaries.json").read_text()).items()}

    return run


def test_llm_mode_fails_fast_when_the_llm_is_unusable(pipeline):
    no_key = TypeError("Could not resolve authentication method. Expected one of api_key, auth_token...")
    with pytest.raises(SystemExit) as exc:
        pipeline("--summarizer", "llm", llm=make_llm(fail_check=no_key))
    assert "not usable" in str(exc.value) and "ANTHROPIC_API_KEY" in str(exc.value)


def test_llm_mode_rejects_an_unknown_model_before_summarizing(pipeline):
    response = httpx.Response(404, request=httpx.Request("GET", "https://api.anthropic.com/v1/models/gpt-4o"))
    not_found = anthropic.NotFoundError("model: gpt-4o", response=response, body=None)
    with pytest.raises(SystemExit, match="EHR_LLM_MODEL"):
        pipeline("--summarizer", "llm", llm=make_llm(fail_check=not_found))


def test_auto_mode_falls_back_to_rules_when_the_llm_is_unusable(pipeline):
    models = pipeline("--summarizer", "auto", llm=make_llm(fail_check=TypeError("Could not resolve authentication")))
    assert len(models) == 15 and set(models.values()) == {RULES_MODEL}


def test_temporary_llm_error_falls_back_for_that_patient_only(pipeline, caplog):
    llm = make_llm(fail_for={"pat-001002": RuntimeError("overloaded (after SDK retries)")})
    with caplog.at_level(logging.INFO, logger="pipeline"):
        models = pipeline("--summarizer", "auto", llm=llm)
    assert models["pat-001002"] == RULES_MODEL
    assert {m for pid, m in models.items() if pid != "pat-001002"} == {"fake-llm"}
    assert "15 generated (fake-llm: 14, rule-based-fallback: 1), 0 cache hits, 0 failed" in caplog.text


def test_rejected_credentials_mid_run_switch_the_remaining_patients(pipeline):
    models = pipeline("--summarizer", "auto", llm=make_llm(fail_from_call=(3, _auth_error())))
    ordered = [models[pid] for pid in sorted(models)]
    assert ordered[:2] == ["fake-llm", "fake-llm"] and set(ordered[2:]) == {RULES_MODEL}


def test_a_patient_that_fails_everywhere_is_skipped_and_the_run_exits_nonzero(pipeline, monkeypatch, tmp_path):
    real = run_pipeline.summarize_bundle

    def broken_for_one(pid, bundle, backend, dq=None):
        if pid == "pat-001003":
            raise KeyError("entry")
        return real(pid, bundle, backend, dq)

    monkeypatch.setattr(run_pipeline, "summarize_bundle", broken_for_one)
    with pytest.raises(SystemExit) as exc:
        pipeline("--summarizer", "auto", llm=make_llm())
    assert exc.value.code == 1
    written = json.loads((tmp_path / "out" / "summaries.json").read_text())
    assert len(written) == 14 and "pat-001003" not in written      # outputs still written for everyone else


def test_llm_mode_transient_error_skips_the_patient_without_fallback(pipeline, tmp_path):
    with pytest.raises(SystemExit) as exc:
        pipeline("--summarizer", "llm", llm=make_llm(fail_for={"pat-001002": RuntimeError("overloaded")}))
    assert exc.value.code == 1
    written = json.loads((tmp_path / "out" / "summaries.json").read_text())
    assert "pat-001002" not in written and set(written[p]["model"] for p in written) == {"fake-llm"}
