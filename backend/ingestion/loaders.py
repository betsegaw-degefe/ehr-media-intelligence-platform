"""Format-specific loaders. Each one turns a source file into a list of *raw* records:
plain dicts with a common set of keys, values still un-normalized strings.

Common raw keys (any may be missing/None):
  source, record_id, mrn, name, given, family, dob, gender,
  encounter_id, doc_type, title, date, text,
  # lab rows only:
  panel, test, loinc, value, unit, ref_range, flag
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

RawRecord = dict[str, Any]

# Field-name aliases seen across the different export shapes.
JSON_ALIASES: dict[str, list[str]] = {
    "record_id": ["record_id", "id", "document_id"],
    "mrn": ["mrn", "patient_id", "MRN", "medical_record_number"],
    "name": ["name", "patient_name", "full_name"],
    "given": ["first_name", "given_name", "given"],
    "family": ["last_name", "family_name", "family", "surname"],
    "dob": ["dob", "birth_date", "birthDate", "date_of_birth"],
    "gender": ["gender", "sex"],
    "encounter_id": ["encounter_id", "encounter", "visit_id"],
    "doc_type": ["document_type", "type", "doc_type"],
    "title": ["title", "description"],
    "date": ["service_date", "date", "effective_date", "created"],
    "text": ["content", "text", "body", "note"],
}


def _pick(d: dict, keys: list[str]) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def load_json(path: Path) -> list[RawRecord]:
    data = json.loads(Path(path).read_text())
    items = data.get("records", data) if isinstance(data, dict) else data
    out: list[RawRecord] = []
    for idx, item in enumerate(items):
        flat = dict(item)
        if isinstance(item.get("patient"), dict):  # nested patient block
            flat.update(item["patient"])
        rec = {key: _pick(flat, aliases) for key, aliases in JSON_ALIASES.items()}
        rec["source"] = f"{Path(path).name}#{idx}"
        rec["kind"] = "document"
        out.append(rec)
    return out


CSV_COLUMNS = {
    "mrn": "MRN", "name": "Patient Name", "dob": "DOB", "gender": "Sex", "date": "Collected",
    "panel": "Panel", "test": "Test", "loinc": "LOINC", "value": "Result", "unit": "Units",
    "ref_range": "Ref Range", "flag": "Flag",
}


def load_lab_csv(path: Path) -> list[RawRecord]:
    out: list[RawRecord] = []
    with Path(path).open(newline="") as f:
        for idx, row in enumerate(csv.DictReader(f), start=2):  # row 1 is the header
            rec = {key: (row.get(col) or None) for key, col in CSV_COLUMNS.items()}
            rec["source"] = f"{Path(path).name}:row{idx}"
            rec["kind"] = "lab_row"
            out.append(rec)
    return out


TXT_HEADER = {
    "PATIENT": "name", "MRN": "mrn", "DOB": "dob", "SEX": "gender", "GENDER": "gender",
    "DATE OF SERVICE": "date", "DATE": "date", "DOCUMENT TYPE": "doc_type", "TITLE": "title",
    "ENCOUNTER": "encounter_id",
}
_HEADER_LINE = re.compile(r"^\s*([A-Z][A-Z /]+?)\s*:\s*(.*)$")


def load_txt(path: Path) -> list[RawRecord]:
    """A scanned/OCR'd note: an optional `KEY: value` header block, a `----` separator, then free text."""
    raw = Path(path).read_text()
    head, sep, body = raw.partition("\n----\n")
    if not sep:  # no separator: treat the whole file as narrative
        head, body = "", raw
    rec: RawRecord = {"source": Path(path).name, "kind": "document",
                      "record_id": "SCAN-" + re.sub(r"[^A-Za-z0-9]+", "-", Path(path).stem).strip("-")}
    for line in head.splitlines():
        m = _HEADER_LINE.match(line)
        if m and m.group(1).strip() in TXT_HEADER:
            rec[TXT_HEADER[m.group(1).strip()]] = m.group(2).strip() or None
    rec["text"] = body
    return [rec]


def load_directory(raw_dir: Path) -> list[RawRecord]:
    """Load every supported file under raw_dir (recursively)."""
    records: list[RawRecord] = []
    for p in sorted(Path(raw_dir).rglob("*")):
        if p.suffix.lower() == ".json":
            records += load_json(p)
        elif p.suffix.lower() == ".csv":
            records += load_lab_csv(p)
        elif p.suffix.lower() == ".txt":
            records += load_txt(p)
    return records
