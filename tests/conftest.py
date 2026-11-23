from __future__ import annotations

import hashlib
import math
import re
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

TODAY = date(2026, 11, 23)  # fixed so 2-digit-year and future-date logic is deterministic


class HashEmbedder:
    """Tiny deterministic bag-of-words embedder so API/search tests run without downloading a model."""

    dim = 256

    def encode(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory) -> Path:
    import generate_data

    d = tmp_path_factory.mktemp("raw")
    generate_data.main(d)
    return d


@pytest.fixture(scope="session")
def dataset(raw_dir):
    from backend.ingestion.pipeline import run_ingestion

    return run_ingestion(raw_dir, today=TODAY)


@pytest.fixture(scope="session")
def bundles(dataset):
    from backend.fhir.mapper import build_bundles

    return build_bundles(dataset)


@pytest.fixture(scope="session")
def populated_db(tmp_path_factory, dataset, bundles) -> Path:
    """SQLite DB with patients, bundles and rule-based summaries — what run_pipeline.py produces."""
    from backend import store
    from backend.fhir.validate import validate_bundle
    from backend.summarize.cache import SummaryCache
    from backend.summarize.summarizer import RulesBackend, summarize_bundle

    db = tmp_path_factory.mktemp("db") / "ehr.db"
    with store.connect(db) as conn:
        store.save_dataset(conn, dataset)
        cache = SummaryCache(conn)
        for pid, b in bundles.items():
            store.save_bundle(conn, pid, b, validate_bundle(b))
            cache.put(summarize_bundle(pid, b, RulesBackend()))
    return db
