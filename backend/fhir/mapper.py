"""Map the clean intermediate representation to HL7 FHIR R4 resources and one Bundle per patient.

Resource mapping
  CleanPatient                     -> Patient
  CleanRecord(category=document)   -> DocumentReference  (narrative as base64 text/plain attachment)
  CleanRecord(category=imaging)    -> DiagnosticReport   (category RAD, conclusion = Impression, presentedForm)
  CleanRecord(category=lab)        -> DiagnosticReport   (category LAB) + Observation per analyte (result[])
  encounter_id on any record       -> Encounter          (one per distinct encounter id)

Every clinical resource carries `subject -> Patient/<id>`, and `encounter`/`context.encounter`
-> Encounter/<id>. Bundles are `type: collection` with absolute fullUrls so relative references resolve.
"""

from __future__ import annotations

import base64
import re
from collections import defaultdict
from datetime import date, datetime, timezone
from typing import Any

from backend.ingestion.models import CleanDataset, CleanPatient, CleanRecord

FHIR_BASE = "https://ehr-media.example.org/fhir"  # example.org is reserved: no real system is referenced
MRN_SYSTEM = "https://ehr-media.example.org/sid/mrn"

LOINC = "http://loinc.org"
V2_0074 = "http://terminology.hl7.org/CodeSystem/v2-0074"  # diagnostic service section
V2_0203 = "http://terminology.hl7.org/CodeSystem/v2-0203"  # identifier type
V3_ACT = "http://terminology.hl7.org/CodeSystem/v3-ActCode"
V3_INTERP = "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation"
OBS_CATEGORY = "http://terminology.hl7.org/CodeSystem/observation-category"
US_CORE_DOC_CAT = "http://hl7.org/fhir/us/core/CodeSystem/us-core-documentreference-category"
UCUM = "http://unitsofmeasure.org"

# LOINC document-type codes for clinical notes.
NOTE_LOINC = {
    "discharge summary": ("18842-5", "Discharge summary"),
    "progress note": ("11506-3", "Progress note"),
    "emergency department note": ("34111-5", "Emergency department Note"),
    "consultation note": ("11488-4", "Consult note"),
    "referral letter": ("57133-1", "Referral note"),
    "history and physical": ("34117-2", "History and physical note"),
}
ENCOUNTER_CLASS = {"discharge summary": ("IMP", "inpatient encounter"),
                   "emergency department note": ("EMER", "emergency")}
INTERPRETATION = {"H": ("H", "High"), "L": ("L", "Low"), "A": ("A", "Abnormal"), "N": ("N", "Normal")}


def fhir_id(s: str) -> str:
    """FHIR ids: [A-Za-z0-9-.]{1,64}."""
    return re.sub(r"[^A-Za-z0-9\-.]", "-", s)[:64]


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def _instant(d: date | None) -> str | None:
    # DocumentReference.date / DiagnosticReport.issued are `instant`: they need a time and a timezone.
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z") if d else None


def _ref(resource_type: str, rid: str, display: str | None = None) -> dict[str, str]:
    r = {"reference": f"{resource_type}/{rid}"}
    if display:
        r["display"] = display
    return r


# --------------------------------------------------------------------------------------
# Individual resources (returned as plain dicts; validated by fhir.resources in validate.py)
# --------------------------------------------------------------------------------------
def patient_resource(p: CleanPatient) -> dict[str, Any]:
    mr_type = {"coding": [{"system": V2_0203, "code": "MR", "display": "Medical record number"}]}
    identifiers = [{"use": "usual", "type": mr_type, "system": MRN_SYSTEM, "value": p.mrn}]
    identifiers += [{"use": "old", "type": mr_type, "system": MRN_SYSTEM, "value": m} for m in p.alternate_mrns]
    res: dict[str, Any] = {
        "resourceType": "Patient", "id": p.patient_id, "active": True, "identifier": identifiers,
        "gender": p.gender,
    }
    if p.given_name or p.family_name:
        name: dict[str, Any] = {"use": "official", "text": p.display_name}
        if p.family_name:
            name["family"] = p.family_name
        if p.given_name:
            name["given"] = p.given_name.split()
        res["name"] = [name]
    if p.birth_date:
        res["birthDate"] = p.birth_date.isoformat()
    return res


def encounter_resource(enc_id: str, patient: CleanPatient, records: list[CleanRecord]) -> dict[str, Any]:
    cls = ("AMB", "ambulatory")
    for r in records:
        if r.doc_type.lower() in ENCOUNTER_CLASS:
            cls = ENCOUNTER_CLASS[r.doc_type.lower()]
            break
    dates = sorted(r.record_date for r in records if r.record_date)
    res: dict[str, Any] = {
        "resourceType": "Encounter", "id": fhir_id(enc_id.lower()), "status": "finished",
        "class": {"system": V3_ACT, "code": cls[0], "display": cls[1]},  # R4: a single Coding
        "identifier": [{"system": f"{FHIR_BASE}/sid/encounter", "value": enc_id}],
        "subject": _ref("Patient", patient.patient_id, patient.display_name),
    }
    if dates:
        res["period"] = {"start": dates[0].isoformat(), "end": dates[-1].isoformat()}
    return res


def document_reference(r: CleanRecord, patient: CleanPatient) -> dict[str, Any]:
    code = NOTE_LOINC.get(r.doc_type.lower())
    doc_type = {"text": r.doc_type}
    if code:
        doc_type["coding"] = [{"system": LOINC, "code": code[0], "display": code[1]}]
    attachment: dict[str, Any] = {"contentType": "text/plain; charset=utf-8", "data": _b64(r.text), "title": r.title}
    if r.record_date:
        attachment["creation"] = r.record_date.isoformat()
    res: dict[str, Any] = {
        "resourceType": "DocumentReference", "id": fhir_id(r.record_id),
        "identifier": [{"system": f"{FHIR_BASE}/sid/source-record", "value": r.record_id}],
        "status": "current", "docStatus": "final", "type": doc_type,
        "category": [{"coding": [{"system": US_CORE_DOC_CAT, "code": "clinical-note", "display": "Clinical Note"}]}],
        "subject": _ref("Patient", patient.patient_id, patient.display_name),
        "description": r.title,
        "content": [{"attachment": attachment}],
    }
    if r.record_date:
        res["date"] = _instant(r.record_date)
    if r.encounter_id:
        res["context"] = {"encounter": [_ref("Encounter", fhir_id(r.encounter_id.lower()))]}
        if r.record_date:
            res["context"]["period"] = {"start": r.record_date.isoformat()}
    return res


def _impression(text: str) -> str:
    m = re.search(r"Impression:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    return (m.group(1) if m else text).strip()


def imaging_report(r: CleanRecord, patient: CleanPatient) -> dict[str, Any]:
    res: dict[str, Any] = {
        "resourceType": "DiagnosticReport", "id": fhir_id(r.record_id),
        "identifier": [{"system": f"{FHIR_BASE}/sid/source-record", "value": r.record_id}],
        "status": "final",
        "category": [{"coding": [{"system": V2_0074, "code": "RAD", "display": "Radiology"}],
                      "text": r.modality or "Imaging"}],
        "code": {"text": r.title},
        "subject": _ref("Patient", patient.patient_id, patient.display_name),
        "conclusion": _impression(r.text),
        "presentedForm": [{"contentType": "text/plain; charset=utf-8", "data": _b64(r.text), "title": r.title}],
    }
    if r.record_date:
        res["effectiveDateTime"] = r.record_date.isoformat()
        res["issued"] = _instant(r.record_date)
    if r.encounter_id:
        res["encounter"] = _ref("Encounter", fhir_id(r.encounter_id.lower()))
    return res


def _quantity(value: str) -> float | None:
    try:
        return float(value.strip())
    except (ValueError, AttributeError):
        return None


def lab_resources(r: CleanRecord, patient: CleanPatient) -> list[dict[str, Any]]:
    report_id = fhir_id(r.record_id)
    observations: list[dict[str, Any]] = []
    for i, x in enumerate(r.results, 1):
        code: dict[str, Any] = {"text": x.test}
        if x.loinc:
            code["coding"] = [{"system": LOINC, "code": x.loinc, "display": x.test}]
        obs: dict[str, Any] = {
            "resourceType": "Observation", "id": fhir_id(f"{report_id}-{i}"), "status": "final",
            "category": [{"coding": [{"system": OBS_CATEGORY, "code": "laboratory", "display": "Laboratory"}]}],
            "code": code, "subject": _ref("Patient", patient.patient_id),
            "interpretation": [{"coding": [{"system": V3_INTERP, "code": INTERPRETATION[x.flag][0],
                                            "display": INTERPRETATION[x.flag][1]}]}],
        }
        q = _quantity(x.value)
        if q is not None:
            obs["valueQuantity"] = {"value": q, **({"unit": x.unit, "system": UCUM, "code": x.unit} if x.unit else {})}
        else:
            obs["valueString"] = x.value
        if x.reference_range:
            obs["referenceRange"] = [{"text": x.reference_range}]
        if r.record_date:
            obs["effectiveDateTime"] = r.record_date.isoformat()
        if r.encounter_id:
            obs["encounter"] = _ref("Encounter", fhir_id(r.encounter_id.lower()))
        observations.append(obs)

    abnormal = [x for x in r.results if x.flag != "N"]
    conclusion = ("Abnormal: " + "; ".join(f"{x.test} {x.value} {x.unit or ''}".strip() + f" ({INTERPRETATION[x.flag][1].lower()})"
                                           for x in abnormal)) if abnormal else "All results within reference range."
    report: dict[str, Any] = {
        "resourceType": "DiagnosticReport", "id": report_id,
        "identifier": [{"system": f"{FHIR_BASE}/sid/source-record", "value": r.record_id}],
        "status": "final",
        "category": [{"coding": [{"system": V2_0074, "code": "LAB", "display": "Laboratory"}]}],
        "code": {"text": r.title},
        "subject": _ref("Patient", patient.patient_id, patient.display_name),
        "result": [_ref("Observation", o["id"], o["code"]["text"]) for o in observations],
        "conclusion": conclusion,
        "presentedForm": [{"contentType": "text/plain; charset=utf-8", "data": _b64(r.text), "title": r.title}],
    }
    if r.record_date:
        report["effectiveDateTime"] = r.record_date.isoformat()
        report["issued"] = _instant(r.record_date)
    if r.encounter_id:
        report["encounter"] = _ref("Encounter", fhir_id(r.encounter_id.lower()))
    return [report, *observations]


# --------------------------------------------------------------------------------------
# Bundles
# --------------------------------------------------------------------------------------
def _entry(res: dict[str, Any]) -> dict[str, Any]:
    return {"fullUrl": f"{FHIR_BASE}/{res['resourceType']}/{res['id']}", "resource": res}


def build_patient_bundle(patient: CleanPatient, records: list[CleanRecord]) -> dict[str, Any]:
    resources: list[dict[str, Any]] = [patient_resource(patient)]
    by_enc: dict[str, list[CleanRecord]] = defaultdict(list)
    for r in records:
        if r.encounter_id:
            by_enc[r.encounter_id].append(r)
    resources += [encounter_resource(e, patient, recs) for e, recs in sorted(by_enc.items())]
    for r in sorted(records, key=lambda x: (x.record_date or date.min, x.record_id)):
        if r.category == "document":
            resources.append(document_reference(r, patient))
        elif r.category == "imaging":
            resources.append(imaging_report(r, patient))
        else:
            resources += lab_resources(r, patient)
    return {
        "resourceType": "Bundle", "id": fhir_id(f"bundle-{patient.patient_id}"), "type": "collection",
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "entry": [_entry(x) for x in resources],
    }


def build_bundles(ds: CleanDataset) -> dict[str, dict[str, Any]]:
    """patient_id -> Bundle dict."""
    by_mrn: dict[str, list[CleanRecord]] = defaultdict(list)
    for r in ds.records:
        by_mrn[r.patient_mrn].append(r)
    return {p.patient_id: build_patient_bundle(p, by_mrn[p.mrn]) for p in ds.patients}
