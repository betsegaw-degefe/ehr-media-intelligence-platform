"""SQLite store for patients, FHIR Bundles, a flattened resource index, audit logs and rejected records.

The Bundle JSON is the source of truth; `resources` is a denormalized read model so the API can list a
patient's records (and the search index can be rebuilt) without re-parsing every Bundle.
"""

from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from backend.ingestion.models import CleanDataset

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    patient_id TEXT PRIMARY KEY, mrn TEXT NOT NULL, name TEXT, given_name TEXT, family_name TEXT,
    birth_date TEXT, gender TEXT, alternate_mrns TEXT, audit TEXT
);
CREATE TABLE IF NOT EXISTS bundles (
    patient_id TEXT PRIMARY KEY REFERENCES patients(patient_id), bundle_json TEXT NOT NULL,
    content_hash TEXT NOT NULL, valid INTEGER NOT NULL, error_count INTEGER NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS resources (
    resource_type TEXT NOT NULL, resource_id TEXT NOT NULL, patient_id TEXT NOT NULL,
    record_date TEXT, title TEXT, subtype TEXT, text TEXT, encounter_id TEXT,
    PRIMARY KEY (resource_type, resource_id)
);
CREATE INDEX IF NOT EXISTS idx_resources_patient ON resources(patient_id);
CREATE TABLE IF NOT EXISTS record_audit (
    record_id TEXT PRIMARY KEY, patient_id TEXT, source TEXT, audit TEXT
);
CREATE TABLE IF NOT EXISTS rejected (source TEXT, reason TEXT, raw TEXT);
"""


@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def bundle_content_hash(bundle: dict[str, Any]) -> str:
    """Hash of the clinical content only (ignores Bundle.timestamp) — the summary cache key depends on it."""
    content = json.dumps(bundle.get("entry", []), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode()).hexdigest()


def decode_attachment(attachment: dict[str, Any]) -> str:
    """Base64 attachment data -> text ("" if missing or not UTF-8)."""
    try:
        return base64.b64decode(attachment.get("data", "")).decode("utf-8")
    except Exception:  # noqa: BLE001 - malformed data should not crash indexing
        return ""


def flatten_bundle(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    """Searchable rows for DocumentReference and DiagnosticReport entries."""
    rows = []
    for e in bundle.get("entry", []):
        r = e["resource"]
        rt = r["resourceType"]
        if rt == "DocumentReference":
            att = r["content"][0]["attachment"]
            enc = (r.get("context", {}).get("encounter") or [{}])[0].get("reference")
            rows.append(dict(resource_type=rt, resource_id=r["id"], record_date=(r.get("date") or "")[:10] or None,
                             title=r.get("description") or att.get("title"), subtype=r["type"].get("text"),
                             text=decode_attachment(att), encounter_id=enc))
        elif rt == "DiagnosticReport":
            cat = r["category"][0]["coding"][0]["code"]
            rows.append(dict(resource_type=rt, resource_id=r["id"], record_date=r.get("effectiveDateTime"),
                             title=r["code"].get("text"),
                             subtype="Imaging" if cat == "RAD" else "Laboratory",
                             text=decode_attachment(r["presentedForm"][0]) if r.get("presentedForm") else r.get("conclusion", ""),
                             encounter_id=(r.get("encounter") or {}).get("reference")))
    return rows


DATASET_TABLES = ("patients", "bundles", "resources", "record_audit", "rejected")


def save_dataset(conn: sqlite3.Connection, ds: CleanDataset) -> None:
    """Replace the stored dataset with this run's. Each pipeline run rebuilds everything from the raw files, so
    patients merged away or records that moved to another patient must not linger from an earlier run. The
    summary cache lives in its own table and is kept. Runs inside the caller's transaction, so readers see
    either the old dataset or the new one, never a mix."""
    for table in DATASET_TABLES:
        conn.execute(f"DELETE FROM {table}")
    for p in ds.patients:
        conn.execute(
            "INSERT OR REPLACE INTO patients VALUES (?,?,?,?,?,?,?,?,?)",
            (p.patient_id, p.mrn, p.display_name, p.given_name, p.family_name,
             p.birth_date.isoformat() if p.birth_date else None, p.gender,
             json.dumps(p.alternate_mrns), json.dumps([a.model_dump() for a in p.audit])))
    pid_by_mrn = {p.mrn: p.patient_id for p in ds.patients}
    for r in ds.records:
        conn.execute("INSERT OR REPLACE INTO record_audit VALUES (?,?,?,?)",
                     (r.record_id, pid_by_mrn[r.patient_mrn], r.source,
                      json.dumps([a.model_dump() for a in r.audit])))
    for rj in ds.rejected:
        conn.execute("INSERT INTO rejected VALUES (?,?,?)", (rj.source, rj.reason, json.dumps(rj.raw)))


def save_bundle(conn: sqlite3.Connection, patient_id: str, bundle: dict[str, Any], validation: dict[str, Any]) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO bundles VALUES (?,?,?,?,?,?)",
        (patient_id, json.dumps(bundle), bundle_content_hash(bundle), int(validation["valid"]),
         len(validation["errors"]), datetime.now(timezone.utc).isoformat(timespec="seconds")))
    conn.execute("DELETE FROM resources WHERE patient_id = ?", (patient_id,))
    for row in flatten_bundle(bundle):
        conn.execute("INSERT INTO resources VALUES (:resource_type,:resource_id,:patient_id,:record_date,"
                     ":title,:subtype,:text,:encounter_id)", {**row, "patient_id": patient_id})


def get_bundle(conn: sqlite3.Connection, patient_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT bundle_json FROM bundles WHERE patient_id = ?", (patient_id,)).fetchone()
    return json.loads(row["bundle_json"]) if row else None


def list_bundles(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT b.*, p.mrn, p.name FROM bundles b JOIN patients p USING (patient_id) "
                        "ORDER BY patient_id").fetchall()


def get_patient(conn: sqlite3.Connection, patient_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["alternate_mrns"] = json.loads(d["alternate_mrns"] or "[]")
    d["audit"] = json.loads(d["audit"] or "[]")
    return d


def list_patients(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute(
        "SELECT patient_id, mrn, name, birth_date, gender FROM patients ORDER BY patient_id")]


def patient_resources(conn: sqlite3.Connection, patient_id: str) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute(
        "SELECT * FROM resources WHERE patient_id = ? ORDER BY record_date DESC, resource_id", (patient_id,))]


def all_resources(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute(
        "SELECT r.*, p.mrn, p.name AS patient_name FROM resources r JOIN patients p USING (patient_id)")]


def record_audit(conn: sqlite3.Connection, record_id: str) -> list[dict[str, Any]]:
    row = conn.execute("SELECT audit FROM record_audit WHERE record_id = ?", (record_id,)).fetchone()
    return json.loads(row["audit"]) if row else []


def record_audits(conn: sqlite3.Connection, record_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Audit trails for many records in one query."""
    if not record_ids:
        return {}
    marks = ",".join("?" * len(record_ids))
    rows = conn.execute(f"SELECT record_id, audit FROM record_audit WHERE record_id IN ({marks})", record_ids)
    return {r["record_id"]: json.loads(r["audit"]) for r in rows}
