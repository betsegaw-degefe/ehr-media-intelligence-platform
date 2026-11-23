"""Task 4: semantic search index and FastAPI endpoints."""

from __future__ import annotations

import time
import uuid

import pytest
from fastapi.testclient import TestClient

from backend import store
from backend.api.main import create_app
from backend.search.index import SearchIndex
from backend.summarize.cache import SummaryCache
from tests.conftest import HashEmbedder


def _build_index(db, embedder) -> SearchIndex:
    index = SearchIndex(embedder, collection=f"test-{uuid.uuid4().hex[:8]}")
    with store.connect(db) as conn:
        cache = SummaryCache(conn)
        summaries = {p["patient_id"]: cache.latest(p["patient_id"]).model_dump() for p in store.list_patients(conn)}
        index.rebuild(store.all_resources(conn), summaries)
    return index


@pytest.fixture(scope="module")
def client(populated_db):
    app = create_app(db_path=populated_db, index=_build_index(populated_db, HashEmbedder()))
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


def test_search_latency_with_real_model_under_two_seconds(populated_db):
    """Spec: < 2 s for 50 records. Skipped when the sentence-transformer model can't be loaded (offline)."""
    try:
        from backend.search.index import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"embedding model unavailable: {e}")
    index = _build_index(populated_db, embedder)
    assert index.count() >= 50
    index.search("warm-up")
    t0 = time.perf_counter()
    results, _ = index.search("pulmonary nodule needing follow-up CT", top_k=5)
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0
    assert results[0]["patient_name"] == "Robert Chen"
