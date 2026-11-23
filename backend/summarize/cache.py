"""SQLite summary cache keyed by patient ID + bundle content hash (+ model and prompt version).

A re-run with unchanged records is a cache hit and makes no API call; any change to a patient's FHIR content
changes the hash and forces a fresh summary for that patient only.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from backend.summarize.summarizer import StoredSummary

SCHEMA = """
CREATE TABLE IF NOT EXISTS summaries (
    patient_id TEXT NOT NULL, record_hash TEXT NOT NULL, model TEXT NOT NULL, prompt_version TEXT NOT NULL,
    summary_json TEXT NOT NULL, created_at TEXT NOT NULL,
    PRIMARY KEY (patient_id, record_hash, model, prompt_version)
);
"""


class SummaryCache:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        conn.executescript(SCHEMA)

    def get(self, patient_id: str, record_hash: str, model: str, prompt_version: str) -> StoredSummary | None:
        row = self.conn.execute(
            "SELECT summary_json FROM summaries WHERE patient_id=? AND record_hash=? AND model=? AND prompt_version=?",
            (patient_id, record_hash, model, prompt_version)).fetchone()
        return StoredSummary.model_validate_json(row[0]) if row else None

    def put(self, s: StoredSummary) -> None:
        self.conn.execute("INSERT OR REPLACE INTO summaries VALUES (?,?,?,?,?,?)",
                          (s.patient_id, s.record_hash, s.model, s.prompt_version, s.model_dump_json(),
                           datetime.now(timezone.utc).isoformat(timespec="seconds")))
        self.conn.commit()

    def latest(self, patient_id: str, record_hash: str | None = None) -> StoredSummary | None:
        """Most recent summary for a patient, preferring an LLM summary over the rule-based fallback."""
        sql = "SELECT summary_json FROM summaries WHERE patient_id=?"
        args: list[str] = [patient_id]
        if record_hash:
            sql += " AND record_hash=?"
            args.append(record_hash)
        sql += " ORDER BY (model = 'rule-based-fallback'), created_at DESC LIMIT 1"
        row = self.conn.execute(sql, args).fetchone()
        return StoredSummary.model_validate_json(row[0]) if row else None
