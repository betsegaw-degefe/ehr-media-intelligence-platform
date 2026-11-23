"""Task 2: FHIR R4 mapping, validation and storage."""

from __future__ import annotations

import copy
import json

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
