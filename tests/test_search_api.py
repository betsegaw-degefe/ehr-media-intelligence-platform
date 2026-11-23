"""Task 4: semantic search index and FastAPI endpoints."""

from __future__ import annotations

import time
import uuid

import pytest
from fastapi.testclient import TestClient

from backend import store
from backend.api.main import create_app
from backend.search.index import RESOURCE_TYPES, SUMMARY_TYPE, SearchIndex
from backend.summarize.cache import SummaryCache
from tests.conftest import HashEmbedder


def _build_index(db, embedder) -> SearchIndex:
    index = SearchIndex(embedder, collection=f"test-{uuid.uuid4().hex[:8]}")
    with store.connect(db) as conn:
        cache = SummaryCache(conn)
        summaries = {p["patient_id"]: cache.latest(p["patient_id"]).model_dump() for p in store.list_patients(conn)}
        index.rebuild(store.all_resources(conn), summaries, store.list_patients(conn))
    return index


@pytest.fixture(scope="module")
def client(summarized_db):
    app = create_app(db_path=summarized_db, index=_build_index(summarized_db, HashEmbedder()))
    with TestClient(app) as c:
        yield c


def test_search_returns_top5_ranked_with_scores(client):
    r = client.post("/search", json={"query": "breast mass mammogram BI-RADS"})
    assert r.status_code == 200
    body = r.json()
    results = body["results"]
    assert len(results) == 5
    scores = [x["score"] for x in results]
    assert scores == sorted(scores, reverse=True) and all(0 <= s <= 1 for s in scores)
    assert results[0]["patient_name"] == "Linda Okafor"
    assert {"mrn", "date", "resource_type", "snippet", "summary_snippet"} <= results[0].keys()


def test_resource_type_filter(client):
    r = client.post("/search", params={"resource_type": "DocumentReference"}, json={"query": "chest x-ray"})
    assert {x["resource_type"] for x in r.json()["results"]} == {"DocumentReference"}


def test_date_range_filter(client):
    r = client.post("/search", params={"date_from": "2025-06-01", "date_to": "2025-07-31"},
                    json={"query": "patient", "top_k": 20})
    dates = [x["date"] for x in r.json()["results"]]
    assert dates and all("2025-06-01" <= d <= "2025-07-31" for d in dates)


def test_invalid_requests(client):
    assert client.post("/search", json={"query": ""}).status_code == 422
    assert client.post("/search", params={"resource_type": "Banana"}, json={"query": "x"}).status_code == 422
    assert client.post("/search", params={"date_from": "2025-08-01", "date_to": "2025-01-01"},
                       json={"query": "x"}).status_code == 422


def test_patient_detail_includes_summary_resources_and_audit(client):
    r = client.get("/patients/pat-001007")
    assert r.status_code == 200
    d = r.json()
    assert d["patient"]["alternate_mrns"] == ["MRN-001070"]
    assert d["summary"]["disclaimer"] and d["summary"]["key_diagnoses"]
    assert len(d["resources"]) == 3
    assert any(a["action"] == "merged" for res in d["resources"] for a in res["audit"])
    assert client.get("/patients/pat-999999").status_code == 404
    assert client.get("/patients/pat-001007/bundle").json()["resourceType"] == "Bundle"


def test_search_latency_with_real_model_under_two_seconds(summarized_db):
    """Spec: < 2 s for 50 records. Skipped when the sentence-transformer model can't be loaded (offline)."""
    try:
        from backend.search.index import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"embedding model unavailable: {e}")
    index = _build_index(summarized_db, embedder)
    assert index.count() >= 50
    index.search("warm-up")
    t0 = time.perf_counter()
    results, _ = index.search("pulmonary nodule needing follow-up CT", top_k=5)
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0
    assert results[0]["patient_name"] == "Robert Chen"


# ---------------------------------------------------------------- code-review regressions
def _row(rid, pid="p1", date=None, title="Note", name="Jane Doe", text="heart failure follow-up"):
    return dict(resource_type="DocumentReference", resource_id=rid, patient_id=pid, mrn="MRN-000001",
                patient_name=name, record_date=date, title=title, subtype="Progress Note", text=text,
                encounter_id=None)


def _summary(pid="p1", chief="Crushing chest pain"):
    return dict(chief_concern=chief, key_diagnoses=["Myocardial infarction"], flagged_anomalies=[],
                narrative="Admitted with chest pain.", model="m")


def _index():
    return SearchIndex(HashEmbedder(), collection=f"test-{uuid.uuid4().hex[:8]}")


def test_date_filter_excludes_undated_records():
    idx = _index()
    idx.rebuild([_row("dated", date="2019-06-01"), _row("undated")], {})
    from datetime import date

    to_only, _ = idx.search("heart failure", 5, date_to=date(2020, 1, 1))
    assert [r["resource_id"] for r in to_only] == ["dated"]
    unfiltered, _ = idx.search("heart failure", 5)
    assert {r["resource_id"] for r in unfiltered} == {"dated", "undated"}


def test_summary_indexed_for_patient_without_records():
    idx = _index()
    patients = [{"patient_id": "p2", "mrn": "MRN-000002", "name": "No Records"}]
    idx.rebuild([_row("r1")], {"p2": _summary("p2")}, patients)
    res, _ = idx.search("chest pain", 5, resource_type=SUMMARY_TYPE)
    assert [(r["patient_id"], r["patient_name"]) for r in res] == [("p2", "No Records")]


def test_missing_name_and_title_are_tolerated():
    idx = _index()
    idx.rebuild([_row("r1", name=None, title=None)], {})
    (hit,), _ = idx.search("heart failure", 5)
    assert hit["patient_name"] == "MRN-000001" and hit["title"] == "Progress Note"
    assert "None" not in idx.collection.get(ids=["DocumentReference/r1"], include=["documents"])["documents"][0]


def test_summary_snippet_keeps_chief_concern():
    idx = _index()
    idx.rebuild([_row("r1")], {"p1": _summary(chief="Crushing chest pain")})
    res, _ = idx.search("chest pain", 5, resource_type=SUMMARY_TYPE)
    assert res[0]["snippet"].startswith("Chief concern: Crushing chest pain")
    rec, _ = idx.search("heart failure", 5, resource_type="DocumentReference")
    assert rec[0]["snippet"] == "heart failure follow-up"  # record snippets are the body, without the title


def test_rebuild_removes_stale_documents():
    idx = _index()
    idx.rebuild([_row("old"), _row("keep")], {})
    idx.rebuild([_row("keep")], {})
    assert idx.collection.get(include=[])["ids"] == ["DocumentReference/keep"]


def test_running_api_sees_rebuild_from_another_process(tmp_path):
    """Code review: the pipeline (another process) used to delete the collection, breaking a running API, and an
    in-place update alone would leave the API serving stale vectors."""
    import subprocess
    import sys

    from tests.conftest import ROOT

    def rebuild_in_subprocess(ids):
        rows = [_row(i, text=f"heart failure {i}") for i in ids]
        code = ("import sys; sys.path.insert(0, %r); from tests.conftest import HashEmbedder; "
                "from backend.search.index import SearchIndex; from pathlib import Path; "
                "SearchIndex(HashEmbedder(), Path(%r)).rebuild(%r, {})" % (str(ROOT), str(tmp_path), rows))
        subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)

    rebuild_in_subprocess(["a"])
    api = SearchIndex(HashEmbedder(), tmp_path)
    assert [r["resource_id"] for r in api.search("heart failure")[0]] == ["a"]
    rebuild_in_subprocess(["b", "c"])
    assert sorted(r["resource_id"] for r in api.search("heart failure")[0]) == ["b", "c"]
    assert api.count() == 2


def test_resource_types_have_one_source_of_truth(client):
    assert client.get("/resource-types").json() == list(RESOURCE_TYPES)
    for t in RESOURCE_TYPES:
        assert client.post("/search", params={"resource_type": t}, json={"query": "x"}).status_code == 200
    schema = client.get("/openapi.json").json()
    enum = next(v for k, v in schema["components"]["schemas"].items() if k == "ResourceType")["enum"]
    assert enum == list(RESOURCE_TYPES)


def test_skip_summaries_keeps_cached_summaries_in_index(raw_dir, tmp_path, monkeypatch):
    """Code review: --skip-summaries used to rebuild the index without any AISummary documents."""
    import sys

    import run_pipeline
    from backend import config
    from backend.search import index as index_mod

    monkeypatch.setattr(config, "DB_PATH", tmp_path / "ehr.db")
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(index_mod, "SentenceTransformerEmbedder", HashEmbedder)

    def run(*args):
        monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--raw-dir", str(raw_dir), *args])
        run_pipeline.main()
        return index_mod.SearchIndex(HashEmbedder(), tmp_path / "chroma").collection.get(include=["metadatas"])

    first = run("--summarizer", "rules")
    second = run("--skip-summaries")
    count = lambda got: sum(m["resource_type"] == SUMMARY_TYPE for m in got["metadatas"])  # noqa: E731
    assert count(first) == count(second) == 15
    assert len(second["ids"]) == len(first["ids"])


def test_index_built_by_older_version_is_still_searchable():
    """An index written before snippets were stored in metadata must not 500 before the pipeline is re-run."""
    idx = _index()
    idx.collection.add(ids=["DocumentReference/r1"], documents=["Note. Progress Note.\nheart failure follow-up"],
                       embeddings=HashEmbedder().encode(["heart failure follow-up"]),
                       metadatas=[{"resource_type": "DocumentReference", "resource_id": "r1", "patient_id": "p1",
                                   "mrn": "MRN-000001", "patient_name": "Jane Doe", "title": "Note",
                                   "subtype": "Progress Note", "date": "2025-01-01", "date_int": 20250101,
                                   "summary_snippet": ""}])
    (hit,), _ = idx.search("heart failure", 5)
    assert hit["snippet"] == "heart failure follow-up"


def test_index_uses_the_same_summary_as_the_patient_page(raw_dir, tmp_path, monkeypatch):
    """Code review: a rules-only run after an LLM run indexed the rule-based summary while the patient page showed
    the LLM one (latest() prefers it)."""
    import sqlite3
    import sys
    from types import SimpleNamespace

    import run_pipeline
    from backend import config
    from backend.search import index as index_mod
    from backend.summarize.cache import SummaryCache
    from backend.summarize.summarizer import RulesBackend

    class FakeLLM(RulesBackend):
        model = "fake-llm"
        client = SimpleNamespace(models=SimpleNamespace(retrieve=lambda model: None))  # startup check passes

        def summarize(self, inp, feedback=None):
            s, _ = super().summarize(inp, feedback)
            return s.model_copy(update={"narrative": "LLM narrative."}), self.model

    monkeypatch.setattr(config, "DB_PATH", tmp_path / "ehr.db")
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(index_mod, "SentenceTransformerEmbedder", HashEmbedder)

    def run(*args):
        monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--raw-dir", str(raw_dir), *args])
        run_pipeline.main()

    monkeypatch.setattr(run_pipeline, "ClaudeBackend", FakeLLM)
    run("--summarizer", "claude")
    run("--summarizer", "rules")
    got = index_mod.SearchIndex(HashEmbedder(), tmp_path / "chroma").collection.get(
        ids=[f"{SUMMARY_TYPE}/pat-001002"], include=["metadatas"])
    page = SummaryCache(sqlite3.connect(tmp_path / "ehr.db"), read_only=True).latest("pat-001002")
    assert page.model == "fake-llm"
    assert got["metadatas"][0]["subtype"] == "fake-llm"


def test_api_is_read_only_and_cors_is_opt_in(summarized_db, monkeypatch):
    """Code review: the patient page ran DDL through SummaryCache(), and CORS allowed every origin."""
    import sqlite3

    from backend import config

    app = create_app(db_path=summarized_db, index=_build_index(summarized_db, HashEmbedder()))
    opened: list[str] = []
    real_connect = sqlite3.connect

    def spy(database, *args, **kwargs):
        opened.append(str(database))
        return real_connect(database, *args, **kwargs)

    with TestClient(app) as c:
        monkeypatch.setattr(sqlite3, "connect", spy)
        r = c.get("/patients/pat-001002", headers={"Origin": "https://evil.example"})
        c.get("/patients")
        monkeypatch.setattr(sqlite3, "connect", real_connect)
    assert r.status_code == 200 and r.json()["summary"] is not None
    assert "access-control-allow-origin" not in r.headers          # no CORS unless configured
    assert opened and all("mode=ro" in d for d in opened)           # every API connection is read-only

    monkeypatch.setattr(config, "CORS_ORIGINS", ["http://localhost:5173"])
    app2 = create_app(db_path=summarized_db, index=_build_index(summarized_db, HashEmbedder()))
    with TestClient(app2) as c:
        r = c.get("/resource-types", headers={"Origin": "http://localhost:5173"})
        assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"
