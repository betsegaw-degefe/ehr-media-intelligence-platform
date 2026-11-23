"""Summary cache: read-only access for the API (code review: a page view must never run DDL)."""

from __future__ import annotations

import sqlite3

import pytest

from backend.summarize.cache import SummaryCache
from backend.summarize.summarizer import PROMPT_VERSION, StoredSummary


def _stored() -> StoredSummary:
    return StoredSummary(chief_concern="c", key_diagnoses=[], recent_media=[], flagged_anomalies=[], narrative="n",
                         confidence="low", confidence_rationale="r", patient_id="p", record_hash="h", notes_hash="n",
                         model="m", requested_model="m", prompt_version=PROMPT_VERSION, generated_at="t", word_count=1)


def test_read_only_never_drops_an_old_schema_table(tmp_path):
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE summaries (patient_id TEXT, record_hash TEXT, model TEXT, prompt_version TEXT, "
                 "summary_json TEXT, created_at TEXT)")
    conn.execute("INSERT INTO summaries VALUES ('p','h','m','v','{}','t')")
    conn.commit()
    reader = SummaryCache(sqlite3.connect(db), read_only=True)
    assert reader.latest("p", "h") is None and reader.get("p", "h", "n", "m") is None
    assert sqlite3.connect(db).execute("SELECT count(*) FROM summaries").fetchone()[0] == 1


def test_read_only_works_on_a_read_only_connection_and_missing_table(tmp_path):
    db = tmp_path / "t.db"
    sqlite3.connect(db).execute("CREATE TABLE other (x)")
    ro = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    assert SummaryCache(ro, read_only=True).latest("p") is None          # no table yet: empty, no error

    writer = sqlite3.connect(db)
    SummaryCache(writer).put(_stored())
    assert SummaryCache(ro, read_only=True).latest("p", "h").chief_concern == "c"
    with pytest.raises(RuntimeError):
        SummaryCache(ro, read_only=True).put(_stored())
