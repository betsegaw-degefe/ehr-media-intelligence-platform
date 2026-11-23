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


def load_json(path: Path, errors: list[tuple[str, str]] | None = None) -> list[RawRecord]:
    """Accepts {"records": [...]}, a bare list of records, or a single record object. Items that aren't
    objects are skipped and reported in `errors` rather than aborting the file."""
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(data, dict) and isinstance(data.get("records"), list):
        items = data["records"]
    elif isinstance(data, list):
        items = data
    elif isinstance(data, dict) and any(k in data for keys in JSON_ALIASES.values() for k in keys):
        items = [data]
    else:
        raise ValueError("no 'records' list and not a single record object")
    out: list[RawRecord] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            if errors is not None:
                errors.append((f"{Path(path).name}#{idx}", f"item is {type(item).__name__}, not an object"))
            continue
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
    with Path(path).open(newline="", encoding="utf-8-sig") as f:  # tolerate Excel's byte-order mark
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
_SEPARATOR = re.compile(r"^\s*[-=_]{3,}\s*$")  # ---, -----, ====, with optional surrounding spaces


def _is_header(line: str) -> bool:
    m = _HEADER_LINE.match(line)
    return bool(m and m.group(1).strip() in TXT_HEADER)


def load_txt(path: Path) -> list[RawRecord]:
    """A scanned/OCR'd note: a `KEY: value` header block, then the narrative. The header ends at a separator
    line (---, -----, ===) or, if there is none, at the first line that isn't a known header field."""
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    sep_at = next((i for i, ln in enumerate(lines) if _SEPARATOR.match(ln)), None)
    if sep_at is not None and all(_is_header(ln) or not ln.strip() for ln in lines[:sep_at]):
        head, body = lines[:sep_at], lines[sep_at + 1:]
    else:
        n = 0
        while n < len(lines) and (_is_header(lines[n]) or (not lines[n].strip() and n == 0)):
            n += 1
        head, body = lines[:n], lines[n:]
    head, body = "\n".join(head), "\n".join(body)
    rec: RawRecord = {"source": Path(path).name, "kind": "document",
                      "record_id": "SCAN-" + re.sub(r"[^A-Za-z0-9]+", "-", Path(path).stem).strip("-")}
    for line in head.splitlines():
        m = _HEADER_LINE.match(line)
        if m and m.group(1).strip() in TXT_HEADER:
            rec[TXT_HEADER[m.group(1).strip()]] = m.group(2).strip() or None
    rec["text"] = body
    return [rec]


LOADERS = {".json": "json", ".csv": "csv", ".txt": "txt"}


def load_directory(raw_dir: Path, errors: list[tuple[str, str]] | None = None) -> list[RawRecord]:
    """Load every supported file under raw_dir (recursively). Hidden files (including macOS `._*` resource
    forks) and directories are skipped. A file that can't be parsed is reported in `errors` and skipped,
    so one bad file doesn't stop the run."""
    errors = errors if errors is not None else []
    records: list[RawRecord] = []
    for p in sorted(Path(raw_dir).rglob("*")):
        kind = LOADERS.get(p.suffix.lower())
        if kind is None or not p.is_file() or any(part.startswith(".") for part in p.relative_to(raw_dir).parts):
            continue
        try:
            if kind == "json":
                records += load_json(p, errors)
            elif kind == "csv":
                records += load_lab_csv(p)
            else:
                records += load_txt(p)
        except (ValueError, UnicodeDecodeError, OSError, csv.Error) as e:  # JSONDecodeError is a ValueError
            errors.append((p.name, f"{type(e).__name__}: {e}"))
    return records
