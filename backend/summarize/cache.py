"""SQLite summary cache.

Key: patient_id + record_hash (Bundle clinical content) + notes_hash (data-quality notes fed to the prompt)
     + requested model + prompt version.
A re-run with unchanged inputs is a cache hit and makes no API call; a change to a patient's FHIR content, to
their data-quality notes, to the model or to the prompt forces a fresh summary for that patient only.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from backend.summarize.summarizer import PROMPT_VERSION, RULES_MODEL, StoredSummary

SCHEMA = """
CREATE TABLE IF NOT EXISTS summaries (
    patient_id TEXT NOT NULL, record_hash TEXT NOT NULL, notes_hash TEXT NOT NULL,
    requested_model TEXT NOT NULL, prompt_version TEXT NOT NULL,
    summary_json TEXT NOT NULL, created_at TEXT NOT NULL,
    PRIMARY KEY (patient_id, record_hash, notes_hash, requested_model, prompt_version)
);
"""


REQUIRED_COLUMNS = {"patient_id", "record_hash", "notes_hash", "requested_model", "prompt_version",
                    "summary_json", "created_at"}


class SummaryCache:
    def __init__(self, conn: sqlite3.Connection, read_only: bool = False):
        """The pipeline opens the cache for writing: it creates the table, and rebuilds it if it has an older
        schema (it is only a cache). Readers such as the API pass read_only=True: no DDL ever runs, so viewing a
        page can't drop or lock anything. A missing or outdated table then just reads as empty."""
        self.conn = conn
        self.read_only = read_only
        cols = {r[1] for r in conn.execute("PRAGMA table_info(summaries)")}
        if read_only:
            self.usable = REQUIRED_COLUMNS <= cols
            return
        if cols and not REQUIRED_COLUMNS <= cols:
            conn.execute("DROP TABLE summaries")
        conn.executescript(SCHEMA)
        self.usable = True

    def get(self, patient_id: str, record_hash: str, notes_hash: str, requested_model: str,
            prompt_version: str = PROMPT_VERSION) -> StoredSummary | None:
        if not self.usable:
            return None
        row = self.conn.execute(
            "SELECT summary_json FROM summaries WHERE patient_id=? AND record_hash=? AND notes_hash=? "
            "AND requested_model=? AND prompt_version=?",
            (patient_id, record_hash, notes_hash, requested_model, prompt_version)).fetchone()
        return StoredSummary.model_validate_json(row[0]) if row else None

    def put(self, s: StoredSummary) -> None:
        if self.read_only:
            raise RuntimeError("SummaryCache opened read-only")
        self.conn.execute("INSERT OR REPLACE INTO summaries VALUES (?,?,?,?,?,?,?)",
                          (s.patient_id, s.record_hash, s.notes_hash, s.requested_model, s.prompt_version,
                           s.model_dump_json(), datetime.now(timezone.utc).isoformat(timespec="seconds")))
        self.conn.commit()

    def latest(self, patient_id: str, record_hash: str | None = None,
               prompt_version: str = PROMPT_VERSION) -> StoredSummary | None:
        """The summary to show for a patient. Only summaries from the current prompt version count, and only for
        `record_hash` (or, if not given, for the most recently summarized Bundle content). Among those, an LLM
        summary is preferred over the rule-based fallback, then the newest."""
        if not self.usable:
            return None
        if record_hash is None:
            row = self.conn.execute(
                "SELECT record_hash FROM summaries WHERE patient_id=? AND prompt_version=? "
                "ORDER BY created_at DESC, rowid DESC LIMIT 1", (patient_id, prompt_version)).fetchone()
            if not row:
                return None
            record_hash = row[0]
        row = self.conn.execute(
            "SELECT summary_json FROM summaries WHERE patient_id=? AND record_hash=? AND prompt_version=? "
            "ORDER BY (requested_model = ?), created_at DESC, rowid DESC LIMIT 1",
            (patient_id, record_hash, prompt_version, RULES_MODEL)).fetchone()
        return StoredSummary.model_validate_json(row[0]) if row else None
