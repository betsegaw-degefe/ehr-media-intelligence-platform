"""Task 2: FHIR R4 mapping, validation and storage."""

from __future__ import annotations

import copy
import json

import pytest

from backend import store
from backend.fhir.validate import validate_bundle, write_report


def _resources(bundle, rtype):
    return [e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == rtype]


def test_every_generated_bundle_is_valid_fhir(bundles):
    results = {pid: validate_bundle(b) for pid, b in bundles.items()}
    errors = {pid: r["errors"] for pid, r in results.items() if r["errors"]}
    assert not errors, json.dumps(errors, indent=2)


def test_bundle_contains_required_resource_types_with_references(bundles):
    b = bundles["pat-001002"]  # heart-failure patient: note + imaging + labs
    types = {e["resource"]["resourceType"] for e in b["entry"]}
    assert {"Patient", "Encounter", "DocumentReference", "DiagnosticReport", "Observation"} <= types
    for res in _resources(b, "DocumentReference") + _resources(b, "DiagnosticReport"):
        assert res["subject"]["reference"] == "Patient/pat-001002"
    dr = next(r for r in _resources(b, "DiagnosticReport") if r["category"][0]["coding"][0]["code"] == "LAB")
    assert dr["result"] and all(ref["reference"].startswith("Observation/") for ref in dr["result"])
    assert dr["encounter"]["reference"].startswith("Encounter/")
    doc = _resources(b, "DocumentReference")[0]
    assert doc["context"]["encounter"][0]["reference"].startswith("Encounter/")
    assert doc["type"]["coding"][0]["code"] == "18842-5"  # LOINC discharge summary


def test_merged_mrn_kept_as_old_identifier(bundles):
    patient = _resources(bundles["pat-001007"], "Patient")[0]
    assert {(i["use"], i["value"]) for i in patient["identifier"]} == {("usual", "MRN-001007"), ("old", "MRN-001070")}


def test_schema_and_reference_errors_are_reported(bundles, tmp_path):
    bad = copy.deepcopy(bundles["pat-001001"])
    patient = _resources(bad, "Patient")[0]
    patient["gender"] = "F"                         # not in the administrative-gender value set
    patient["birthDate"] = "14/03/1958"              # not a FHIR date
    doc = _resources(bad, "DocumentReference")[0]
    doc["subject"]["reference"] = "Patient/missing"  # dangling reference
    del doc["content"]                               # required (1..*)
    result = validate_bundle(bad)
    assert not result["valid"]
    paths = {(e["resource"].split("/")[0], e["path"].split(".")[0]) for e in result["errors"]}
    assert ("Patient", "gender") in paths
    assert ("Patient", "birthDate") in paths
    assert ("DocumentReference", "content") in paths
    assert any("unresolved reference 'Patient/missing'" in e["message"] for e in result["errors"])

    json_path, md_path = write_report({"pat-001001": result}, tmp_path)
    assert json.loads(json_path.read_text())["invalid"] == 1
    assert "❌" in md_path.read_text() and "Patient/missing" in md_path.read_text()


def test_bundles_round_trip_through_sqlite(populated_db, bundles):
    with store.connect(populated_db) as conn:
        assert len(store.list_bundles(conn)) == 15
        b = store.get_bundle(conn, "pat-001005")
        assert b["entry"] == bundles["pat-001005"]["entry"]
        rows = store.patient_resources(conn, "pat-001005")
        assert {r["resource_type"] for r in rows} == {"DocumentReference", "DiagnosticReport"}
        assert any("BI-RADS 4C" in r["text"] for r in rows)


def test_content_hash_ignores_bundle_timestamp(bundles):
    b = copy.deepcopy(bundles["pat-001001"])
    h = store.bundle_content_hash(b)
    b["timestamp"] = "2030-01-01T00:00:00Z"
    assert store.bundle_content_hash(b) == h
    _resources(b, "Patient")[0]["gender"] = "male"
    assert store.bundle_content_hash(b) != h


def test_duplicate_resource_ids_are_reported(bundles):
    bad = copy.deepcopy(bundles["pat-001002"])
    bad["entry"].append(copy.deepcopy(next(e for e in bad["entry"]
                                           if e["resource"]["resourceType"] == "DiagnosticReport")))
    result = validate_bundle(bad)
    assert not result["valid"]
    assert any("duplicate resource id" in e["message"] for e in result["errors"])


def test_critical_lab_flag_maps_to_hl7_interpretation(dataset):
    from backend.fhir.mapper import build_patient_bundle

    patient = dataset.patients[0]
    panel = next(r for r in dataset.records if r.patient_mrn == patient.mrn and r.category == "lab")
    critical = panel.model_copy(update={"results": [panel.results[0].model_copy(update={"flag": "HH"})]})
    bundle = build_patient_bundle(patient, [critical])
    assert validate_bundle(bundle)["valid"]
    obs = next(e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == "Observation")
    assert obs["interpretation"][0]["coding"][0]["code"] == "HH"


def test_rerun_replaces_dataset_when_records_move_between_patients(tmp_path):
    """Code review: a record that moves to another patient in a later run used to crash save_bundle
    (UNIQUE constraint on resources) and leave the merged-away patient behind."""
    from datetime import date

    from backend.fhir.mapper import build_bundles
    from backend.ingestion.pipeline import clean_raw_records

    def doc(rid, mrn, dob="1958-03-14", d="2025-02-10"):
        return dict(kind="document", source="ehr_export.json#0", record_id=rid, mrn=mrn, name="Maria Gonzalez",
                    dob=dob, gender="F", date=d, doc_type="Progress Note", title="Note", text=f"note {rid}")

    def save(ds):
        with store.connect(tmp_path / "t.db") as conn:
            store.save_dataset(conn, ds)
            for pid, b in build_bundles(ds).items():
                store.save_bundle(conn, pid, b, validate_bundle(b))

    today = date(2026, 11, 23)
    save(clean_raw_records([doc("a", "1001"), doc("b", "1010", dob="1958-03-15")], today=today))
    save(clean_raw_records([doc("a", "1001"), doc("b", "1001", d="2025-03-01")], today=today))
    with store.connect(tmp_path / "t.db") as conn:
        assert [p["patient_id"] for p in store.list_patients(conn)] == ["pat-001001"]
        assert {r["patient_id"] for r in store.all_resources(conn)} == {"pat-001001"}
        assert [r["patient_id"] for r in store.list_bundles(conn)] == ["pat-001001"]


def test_record_audits_batch_lookup(populated_db):
    with store.connect(populated_db) as conn:
        ids = [r["resource_id"] for r in store.patient_resources(conn, "pat-001007")]
        batch = store.record_audits(conn, ids)
        assert batch == {i: store.record_audit(conn, i) for i in ids if store.record_audit(conn, i)}
        assert store.record_audits(conn, []) == {}


# ---------------------------------------------------------------- PR #3 code-review regressions
def _clean(*raw_records):
    from datetime import date

    from backend.ingestion.pipeline import clean_raw_records

    return clean_raw_records(list(raw_records), today=date(2026, 11, 23))


def _doc(rid="A", text="note", **kw):
    base = dict(kind="document", source="ehr_export.json#0", record_id=rid, mrn="1001", name="Maria Gonzalez",
                dob="1958-03-14", gender="F", date="2025-02-10", doc_type="Progress Note", title="Note", text=text)
    base.update(kw)
    return base


def _lab(**kw):
    base = dict(kind="lab_row", source="lab_results.csv:row2", mrn="1001", name="Maria Gonzalez", dob="1958-03-14",
                gender="F", date="2025-02-10", panel="Chem", test="Potassium", loinc="2823-3", value="4.1",
                unit="mmol/L", ref_range="3.5-5.1", flag="")
    base.update(kw)
    return base


def _bundle(ds):
    from backend.fhir.mapper import build_bundles

    return next(iter(build_bundles(ds).values()))


@pytest.mark.parametrize("bad_entry,message", [
    ({"resource": {"resourceType": "Banana", "id": "x"}}, "unknown resourceType 'Banana'"),
    ({"resource": {"id": "x"}}, "missing resourceType"),
    ({"resource": "not-an-object"}, "not an object"),
])
def test_bad_resource_types_are_reported_not_raised(bundles, bad_entry, message):
    bad = copy.deepcopy(bundles["pat-001001"])
    bad["entry"].append(bad_entry)
    result = validate_bundle(bad)
    assert not result["valid"] and any(message in e["message"] for e in result["errors"])


@pytest.mark.parametrize("value,expected_key", [("", "dataAbsentReason"), ("nan", "valueString"),
                                                ("inf", "valueString"), ("positive", "valueString"),
                                                ("4.1", "valueQuantity")])
def test_lab_values_always_produce_valid_fhir_and_json(value, expected_key):
    b = _bundle(_clean(_lab(value=value)))
    obs = next(e["resource"] for e in b["entry"] if e["resource"]["resourceType"] == "Observation")
    assert expected_key in obs
    assert validate_bundle(b)["valid"]
    json.dumps(b, allow_nan=False)  # NaN/Infinity would raise here


@pytest.mark.parametrize("text,expected", [
    ("Exam: CT\nFindings: Nodule.\nImpression: 8 mm nodule.\nRecommendation: CT in 3 months.\nSigned: Dr X",
     "8 mm nodule."),
    ("Exam: CT\nFindings: Small effusion.\nImpression:\n", "Small effusion."),
    ("Exam: CT\nImpression:   \n", None),
])
def test_imaging_conclusion_is_the_impression_section_only(text, expected):
    b = _bundle(_clean(_doc(text=text, doc_type="Imaging Report - CT")))
    dr = next(e["resource"] for e in b["entry"] if e["resource"]["resourceType"] == "DiagnosticReport")
    assert dr.get("conclusion") == expected
    assert validate_bundle(b)["valid"]


def test_encounter_ids_differing_in_case_are_one_encounter():
    b = _bundle(_clean(_doc("A", encounter_id="ENC-7"), _doc("B", text="two", encounter_id="enc-7")))
    encs = [e["resource"] for e in b["entry"] if e["resource"]["resourceType"] == "Encounter"]
    assert [e["id"] for e in encs] == ["enc-7"]
    assert {i["value"] for i in encs[0]["identifier"]} == {"ENC-7", "enc-7"}
    assert validate_bundle(b)["valid"]


def test_encounter_class_does_not_depend_on_record_order():
    from backend.fhir.mapper import build_patient_bundle

    ds = _clean(_doc("A", doc_type="Emergency Department Note", encounter_id="ENC-1"),
                _doc("B", text="two", doc_type="Discharge Summary", encounter_id="ENC-1"))
    p, recs = ds.patients[0], ds.records

    def enc_class(records):
        b = build_patient_bundle(p, records)
        return next(e["resource"]["class"]["code"] for e in b["entry"] if e["resource"]["resourceType"] == "Encounter")

    assert enc_class(recs) == enc_class(list(reversed(recs))) == "IMP"
    b1, b2 = build_patient_bundle(p, recs), build_patient_bundle(p, list(reversed(recs)))
    assert store.bundle_content_hash(b1) == store.bundle_content_hash(b2)


def test_observation_ids_stay_unique_even_for_64_char_report_ids(dataset):
    from backend.fhir.mapper import build_patient_bundle

    ds = _clean(_lab(test="A"), _lab(test="B", value="5"))
    long_id = ds.records[0].model_copy(update={"record_id": "L" * 64})
    b = build_patient_bundle(ds.patients[0], [long_id])
    obs = [e["resource"]["id"] for e in b["entry"] if e["resource"]["resourceType"] == "Observation"]
    assert len(set(obs)) == 2 and validate_bundle(b)["valid"]
