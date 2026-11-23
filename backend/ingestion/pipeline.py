"""Ingestion pipeline: raw files -> normalized, de-duplicated, identity-resolved CleanDataset.

Steps
  1. load      every JSON / CSV / TXT file into raw dicts                        (loaders.py)
  2. normalize each field (MRN, name, DOB, gender, dates, text) with an audit    (cleaning.py)
  3. reject    records with no usable patient identity (no MRN and no name+DOB)
  4. resolve   identity: fill missing MRNs from name+DOB; merge MRNs that share
               the same name+DOB (likely typos); pick majority demographics per
               patient and flag conflicts
  5. dedupe    documents and lab rows by content fingerprint
  6. group     lab rows into panels (one DiagnosticReport per patient/date/panel)
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from backend.ingestion import cleaning as C
from backend.ingestion.loaders import load_directory
from backend.ingestion.models import (
    AuditEntry, CleanDataset, CleanPatient, CleanRecord, LabResult, RejectedRecord,
)

IMAGING_HINTS = ("imaging", "radiology")  # matched against the source document type


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def _patient_id(mrn: str) -> str:
    return "pat-" + mrn.split("-")[1]


# --------------------------------------------------------------------------------------
# Step 2: per-record normalization into a "staged" dict
# --------------------------------------------------------------------------------------
def _stage(raw: dict[str, Any], today: date) -> dict[str, Any]:
    audit: list[AuditEntry] = []
    mrn, a = C.normalize_mrn(raw.get("mrn")); audit += a
    given, family, a = C.normalize_name(raw.get("name"), raw.get("given"), raw.get("family")); audit += a
    dob, a = C.parse_date(raw.get("dob"), "birth_date", is_birth_date=True, today=today); audit += a
    gender, a = C.normalize_gender(raw.get("gender")); audit += a
    rec_date, a = C.parse_date(raw.get("date"), "record_date", today=today)
    staged = dict(raw, mrn=mrn, given=given, family=family, dob=dob, gender=gender)

    if raw.get("kind") == "document":
        text, ta = C.normalize_text(raw.get("text"))
        if rec_date is None:
            inferred = C.date_from_text(text)
            if inferred:
                a = [AuditEntry(field="record_date", action="filled", new=inferred.isoformat(),
                                reason="date missing from header; inferred from first date in narrative")]
                rec_date = inferred
        audit += a + ta
        staged["text"] = text
    else:
        audit += a
    staged["record_date"] = rec_date
    staged["audit"] = audit
    return staged


def _source_authority(s: dict[str, Any]) -> int:
    """The EHR export is the system of record for identity; OCR'd scans and lab feeds are less reliable."""
    src = s["source"].lower()
    return 3 if ".json" in src else 2 if ".txt" in src else 1


def _demo_key(s: dict[str, Any]) -> tuple | None:
    if s.get("family") and s.get("given") and s.get("dob"):
        return (s["family"].lower(), s["given"].lower(), s["dob"])
    return None


# --------------------------------------------------------------------------------------
# Step 4: identity resolution
# --------------------------------------------------------------------------------------
def _resolve_identities(staged: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    # Which MRNs have been seen with each exact (family, given, dob)?
    mrns_by_demo: dict[tuple, Counter] = defaultdict(Counter)
    for s in staged:
        if s["mrn"] and (k := _demo_key(s)):
            mrns_by_demo[k][s["mrn"]] += 1

    # Best source authority each MRN was seen in (EHR export > scanned notes > lab feed).
    authority: dict[str, int] = defaultdict(int)
    for s in staged:
        if s["mrn"]:
            authority[s["mrn"]] = max(authority[s["mrn"]], _source_authority(s))

    # The same person under several MRNs (e.g. a transposed digit): merge into the MRN from the most
    # authoritative source, breaking ties by record count, then lowest MRN.
    alias: dict[str, str] = {}
    for k, counter in mrns_by_demo.items():
        if len(counter) > 1:
            primary = sorted(counter.items(), key=lambda kv: (-authority[kv[0]], -kv[1], kv[0]))[0][0]
            for other in counter:
                if other != primary:
                    alias[other] = primary

    by_mrn: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in staged:
        if s["mrn"] is None and (k := _demo_key(s)) and k in mrns_by_demo:
            filled = mrns_by_demo[k].most_common(1)[0][0]
            filled = alias.get(filled, filled)
            s["mrn"] = filled
            s["audit"].append(AuditEntry(field="mrn", action="filled", new=filled,
                                         reason="MRN missing; matched another record on exact name + DOB"))
        elif s["mrn"] in alias:
            s["audit"].append(AuditEntry(
                field="mrn", action="merged", old=s["mrn"], new=alias[s["mrn"]],
                reason="conflicting MRN for identical name + DOB; merged into MRN with the most records"))
            s["mrn"] = alias[s["mrn"]]
        by_mrn[s["mrn"]].append(s)
    return by_mrn


def _majority(values: list[Any]) -> tuple[Any, set]:
    present = [v for v in values if v not in (None, "unknown")]
    if not present:
        return (values[0] if values else None), set()
    counts = Counter(present)
    best = max(counts.items(), key=lambda kv: (kv[1], -present.index(kv[0])))[0]  # ties -> first seen
    return best, set(present)


def _build_patient(mrn: str, recs: list[dict[str, Any]], aliases: set[str]) -> CleanPatient:
    audit: list[AuditEntry] = []
    out: dict[str, Any] = {}
    for fld in ("given", "family", "dob", "gender"):
        val, distinct = _majority([r[fld] for r in recs])
        if len(distinct) > 1:
            audit.append(AuditEntry(
                field=fld, action="conflict", old=", ".join(sorted(map(str, distinct))), new=str(val),
                reason=f"{len(distinct)} different values across {len(recs)} source records; kept majority value"))
        if any(r[fld] in (None, "unknown") for r in recs) and val not in (None, "unknown"):
            audit.append(AuditEntry(field=fld, action="filled", new=str(val),
                                    reason="missing on some source records; filled from other records"))
        out[fld] = val
    return CleanPatient(
        patient_id=_patient_id(mrn), mrn=mrn, given_name=out["given"], family_name=out["family"],
        birth_date=out["dob"], gender=out["gender"] or "unknown",
        alternate_mrns=sorted(aliases), audit=audit,
    )


# --------------------------------------------------------------------------------------
# Steps 5-6: documents and lab panels
# --------------------------------------------------------------------------------------
def _encounter_for(mrn: str, d: date | None, given: str | None) -> tuple[str | None, list[AuditEntry]]:
    if given:
        return given, []
    if d is None:
        return None, []
    enc = f"ENC-{mrn.split('-')[1].lstrip('0')}-{d.strftime('%Y%m%d')}"
    return enc, [AuditEntry(field="encounter_id", action="filled", new=enc,
                            reason="no encounter in source; derived from patient + service date")]


def _document_record(s: dict[str, Any]) -> CleanRecord:
    doc_type = (s.get("doc_type") or "Clinical Note").strip()
    is_imaging = any(h in doc_type.lower() for h in IMAGING_HINTS)
    modality = doc_type.split(" - ", 1)[1] if " - " in doc_type else None
    enc, a = _encounter_for(s["mrn"], s["record_date"], s.get("encounter_id"))
    title = s.get("title") or doc_type
    return CleanRecord(
        record_id=s.get("record_id") or f"DOC-{C.content_fingerprint(s['text'])}",
        patient_mrn=s["mrn"], category="imaging" if is_imaging else "document",
        doc_type=doc_type, title=title, record_date=s["record_date"], encounter_id=enc,
        text=s["text"], modality=modality, source=s["source"], audit=s["audit"] + a,
    )


def _lab_records(rows: list[dict[str, Any]], stats: Counter) -> list[CleanRecord]:
    seen: dict[str, dict[str, Any]] = {}
    panels: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        fp = C.content_fingerprint(r["mrn"], str(r["record_date"]), r.get("test"), r.get("value"))
        if fp in seen:
            stats["duplicate_lab_rows_removed"] += 1
            seen[fp]["audit"].append(AuditEntry(field="record", action="deduplicated", old=r["source"],
                                                reason="identical lab result (patient, date, test, value)"))
            continue
        seen[fp] = r
        panels[(r["mrn"], r["record_date"], r.get("panel") or "Lab Results")].append(r)

    out: list[CleanRecord] = []
    for (mrn, d, panel), items in panels.items():
        results = [LabResult(test=i.get("test") or "Unknown test", loinc=i.get("loinc"), value=i.get("value") or "",
                             unit=i.get("unit"), reference_range=i.get("ref_range"),
                             flag=C.normalize_flag(i.get("flag"))) for i in items]
        flag_word = {"H": " [HIGH]", "L": " [LOW]", "A": " [ABNORMAL]", "N": ""}
        lines = [f"Lab panel: {panel}"] + [
            f"- {x.test}: {x.value} {x.unit or ''} (ref {x.reference_range or 'n/a'}){flag_word[x.flag]}".replace("  ", " ")
            for x in results
        ]
        enc, a = _encounter_for(mrn, d, None)
        audit = [e for i in items for e in i["audit"]] + a
        out.append(CleanRecord(
            record_id=f"LAB-{mrn.split('-')[1]}-{d.strftime('%Y%m%d') if d else 'nodate'}-{_slug(panel)}",
            patient_mrn=mrn, category="lab", doc_type="Lab Panel", title=panel, record_date=d,
            encounter_id=enc, text="\n".join(lines), results=results,
            source=", ".join(i["source"] for i in items), audit=audit,
        ))
    return out


# --------------------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------------------
def clean_raw_records(raw_records: list[dict[str, Any]], today: date | None = None) -> CleanDataset:
    today = today or date.today()
    stats: Counter = Counter(raw_records=len(raw_records))
    staged = [_stage(r, today) for r in raw_records]

    rejected: list[RejectedRecord] = []
    usable: list[dict[str, Any]] = []
    for s in staged:
        if s["mrn"] is None and _demo_key(s) is None:
            rejected.append(RejectedRecord(source=s["source"], reason="no MRN and no name + date of birth; "
                                           "cannot be attributed to a patient", raw={k: str(v) for k, v in s.items()
                                                                                    if k not in ("audit",)}))
        else:
            usable.append(s)

    by_mrn = _resolve_identities(usable)

    # Records still without an MRN had a name+DOB no other record shares: reject rather than guess.
    for s in by_mrn.pop(None, []):
        rejected.append(RejectedRecord(source=s["source"], reason="no MRN and name + DOB matched no known patient",
                                       raw={k: str(v) for k, v in s.items() if k != "audit"}))

    patients: list[CleanPatient] = []
    records: list[CleanRecord] = []
    for mrn in sorted(by_mrn):
        recs = by_mrn[mrn]
        aliases = {a.old for r in recs for a in r["audit"] if a.action == "merged" and a.field == "mrn" and a.old}
        patients.append(_build_patient(mrn, recs, aliases))

        docs: dict[str, CleanRecord] = {}
        for s in (r for r in recs if r["kind"] == "document"):
            rec = _document_record(s)
            fp = C.content_fingerprint(mrn, str(rec.record_date), rec.category, rec.text)
            if fp in docs:
                stats["duplicate_documents_removed"] += 1
                docs[fp].audit.append(AuditEntry(field="record", action="deduplicated", old=s["source"],
                                                 reason="same patient, date, type and text (ignoring whitespace)"))
                continue
            docs[fp] = rec
        records += docs.values()
        records += _lab_records([r for r in recs if r["kind"] == "lab_row"], stats)

    stats.update(patients=len(patients), records=len(records), rejected=len(rejected),
                 audit_entries=sum(len(r.audit) for r in records) + sum(len(p.audit) for p in patients))
    return CleanDataset(patients=patients, records=records, rejected=rejected, stats=dict(stats))


def run_ingestion(raw_dir: Path, today: date | None = None) -> CleanDataset:
    return clean_raw_records(load_directory(raw_dir), today=today)
