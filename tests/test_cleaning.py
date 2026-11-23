"""Task 1: ingestion & cleaning edge cases."""

from __future__ import annotations

from datetime import date

import pytest

from backend.ingestion import cleaning as C
from backend.ingestion.loaders import load_txt
from backend.ingestion.pipeline import clean_raw_records
from tests.conftest import TODAY


def raw(**kw):
    base = dict(kind="document", source="test", record_id="R1", mrn="1001", name="Maria Gonzalez",
                dob="1958-03-14", gender="F", date="2025-02-10", doc_type="Progress Note", title="Note",
                text="Chief Complaint: test.\n\nAssessment:\n1. Hypertension")
    base.update(kw)
    return base


# ---------------------------------------------------------------- field-level cleaning
@pytest.mark.parametrize("value", ["1958-03-14", "03/14/1958", "14 Mar 1958", "March 14, 1958", "19580314",
                                   "1958-03-14T09:30:00Z", "03/14/58"])
def test_inconsistent_date_formats_normalize_to_same_iso_date(value):
    d, audit = C.parse_date(value, "birth_date", is_birth_date=True, today=TODAY)
    assert d == date(1958, 3, 14)
    if value != "1958-03-14":
        assert audit and audit[-1].new == "1958-03-14"


def test_two_digit_birth_year_in_the_future_is_shifted_back_a_century():
    d, audit = C.parse_date("03/14/58", "birth_date", is_birth_date=True, today=TODAY)
    assert d == date(1958, 3, 14)
    assert "future birth date" in audit[0].reason


def test_unparseable_and_missing_dates_are_dropped_with_audit():
    assert C.parse_date("not a date", "record_date")[0] is None
    d, audit = C.parse_date(None, "record_date")
    assert d is None and audit[0].action == "missing"


@pytest.mark.parametrize("raw_value,expected", [("F", "female"), ("2", "female"), ("Female ", "female"),
                                                ("m ", "male"), ("1", "male"), ("X", "other"),
                                                ("", "unknown"), (None, "unknown"), ("zzz", "unknown")])
def test_gender_codes_map_to_fhir_administrative_gender(raw_value, expected):
    assert C.normalize_gender(raw_value)[0] == expected


@pytest.mark.parametrize("value", ["MRN-001001", "1001", "001001", "mrn 1001", " 1001 ", "MRN00001001"])
def test_mrn_formats_collapse_to_canonical(value):
    assert C.normalize_mrn(value)[0] == "MRN-001001"


@pytest.mark.parametrize("value", [None, "", "MRN-", "0000", "12345678"])
def test_invalid_mrns_are_rejected(value):
    assert C.normalize_mrn(value)[0] is None


def test_name_formats():
    assert C.normalize_name("GONZALEZ, MARIA")[:2] == ("Maria", "Gonzalez")
    assert C.normalize_name("Maria Gonzalez")[:2] == ("Maria", "Gonzalez")
    assert C.normalize_name(given="KEVIN", family="O'BRIEN")[:2] == ("Kevin", "O'Brien")


# ---------------------------------------------------------------- record-level cleaning
def test_exact_and_whitespace_only_duplicates_are_removed():
    recs = [raw(record_id="A"), raw(record_id="B"),
            raw(record_id="C", text="  Chief Complaint: test.\n\n\n\nAssessment:\n1. Hypertension  ")]
    ds = clean_raw_records(recs, today=TODAY)
    assert len(ds.records) == 1
    assert ds.stats["duplicate_documents_removed"] == 2
    assert sum(a.action == "deduplicated" for a in ds.records[0].audit) == 2


def test_conflicting_mrns_for_same_person_are_merged():
    recs = [raw(record_id="A"), raw(record_id="B", date="2025-03-01", text="other"),
            raw(record_id="C", mrn="1010", date="2025-04-01", text="typo mrn")]  # transposed MRN, same name+DOB
    ds = clean_raw_records(recs, today=TODAY)
    assert len(ds.patients) == 1
    p = ds.patients[0]
    assert p.mrn == "MRN-001001" and p.alternate_mrns == ["MRN-001010"]
    moved = next(r for r in ds.records if r.record_id == "C")
    assert moved.patient_mrn == "MRN-001001"
    assert any(a.action == "merged" for a in moved.audit)


def test_missing_mrn_is_filled_from_name_and_dob_and_unattributable_records_rejected():
    recs = [raw(record_id="A"),
            raw(record_id="B", mrn=None, name="GONZALEZ, MARIA", dob="03/14/1958", text="filled"),
            raw(record_id="C", mrn=None, name=None, dob=None, text="orphan fax page")]
    ds = clean_raw_records(recs, today=TODAY)
    b = next(r for r in ds.records if r.record_id == "B")
    assert b.patient_mrn == "MRN-001001"
    assert any(a.action == "filled" and a.field == "mrn" for a in b.audit)
    assert [r.raw["record_id"] for r in ds.rejected] == ["C"]


def test_conflicting_demographics_keep_majority_and_flag_conflict():
    recs = [raw(record_id="A"), raw(record_id="B", text="x"), raw(record_id="C", name="Mary Gonzalez", text="y"),
            raw(record_id="D", gender=None, text="z")]
    p = clean_raw_records(recs, today=TODAY).patients[0]
    assert p.given_name == "Maria"
    assert p.gender == "female"
    assert any(a.field == "given" and a.action == "conflict" for a in p.audit)
    assert any(a.field == "gender" and a.action == "filled" for a in p.audit)


def test_missing_note_date_is_inferred_from_narrative(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("PATIENT: Doe, Jane\nMRN: 77\nDOB: 1990-01-01\n----\nSeen in clinic on August 26, 2025.\nStable.")
    ds = clean_raw_records(load_txt(f), today=TODAY)
    r = ds.records[0]
    assert r.record_date == date(2025, 8, 26)
    assert any(a.field == "record_date" and a.action == "filled" for a in r.audit)


def test_lab_rows_grouped_into_panels_and_duplicate_rows_removed():
    row = dict(kind="lab_row", source="labs.csv", mrn="1001", name="Maria Gonzalez", dob="1958-03-14", gender="F",
               date="2025-02-10", panel="Diabetes Panel", test="Hemoglobin A1c", loinc="4548-4", value="9.4",
               unit="%", ref_range="4.0-5.6", flag="H")
    ds = clean_raw_records([row, dict(row), dict(row, test="Glucose", loinc="2345-7", value="212")], today=TODAY)
    assert len(ds.records) == 1
    panel = ds.records[0]
    assert panel.category == "lab" and len(panel.results) == 2
    assert ds.stats["duplicate_lab_rows_removed"] == 1
    assert "[HIGH]" in panel.text


# ---------------------------------------------------------------- whole synthetic dataset
def test_synthetic_dataset_end_to_end(dataset):
    assert len(dataset.patients) == 15
    assert len(dataset.rejected) == 1
    assert len(dataset.records) >= 50
    nguyen = next(p for p in dataset.patients if p.family_name == "Nguyen")
    assert nguyen.alternate_mrns == ["MRN-001070"]
    whitfield = next(p for p in dataset.patients if p.family_name == "Whitfield")
    assert whitfield.given_name == "James"  # majority beats the "Jim" variant
    assert all(p.birth_date and p.birth_date.year < 2010 for p in dataset.patients)
    assert all(r.audit is not None and r.record_date for r in dataset.records)


# ---------------------------------------------------------------- code-review regressions
def lab(**kw):
    base = dict(kind="lab_row", source="lab_results.csv:row2", mrn="1001", name="Maria Gonzalez", dob="1958-03-14",
                gender="F", date="2025-02-10", panel="Lipid Panel", test="LDL", loinc=None, value="100", unit="mg/dL",
                ref_range=None, flag="")
    base.update(kw)
    return base


def test_ehr_export_outranks_many_lab_rows_for_demographics():
    recs = [raw(source="ehr_export.json#0")] + [
        lab(dob="1985-03-14", test=f"T{i}", value=str(i), source=f"lab_results.csv:row{i}") for i in range(4)]
    p = clean_raw_records(recs, today=TODAY).patients[0]
    assert p.birth_date == date(1958, 3, 14)
    assert any(a.field == "dob" and a.action == "conflict" for a in p.audit)


def test_transitive_mrn_aliases_resolve_to_one_patient():
    recs = [raw(record_id="a1", mrn="1100", text="a1"),
            raw(record_id="a2", mrn="1010", text="a2"), raw(record_id="a3", mrn="1010", text="a3"),
            raw(record_id="b1", mrn="1010", name="Maria Gonzales", text="b1"),
            raw(record_id="b2", mrn="1001", name="Maria Gonzales", text="b2"),
            raw(record_id="b3", mrn="1001", name="Maria Gonzales", text="b3"),
            raw(record_id="b4", mrn="1001", name="Maria Gonzales", text="b4")]
    ds = clean_raw_records(recs, today=TODAY)
    assert len(ds.patients) == 1
    p = ds.patients[0]
    # Same source authority everywhere; 1010 and 1001 tie on 3 records each, so the lowest MRN wins.
    assert p.mrn == "MRN-001001" and p.alternate_mrns == ["MRN-001010", "MRN-001100"]
    assert {r.patient_mrn for r in ds.records} == {"MRN-001001"}


def test_panel_names_differing_in_case_form_one_panel():
    ds = clean_raw_records([lab(panel="Lipid Panel"), lab(panel="LIPID PANEL", test="HDL", value="50")], today=TODAY)
    assert len(ds.records) == 1 and len(ds.records[0].results) == 2


def test_reused_source_record_ids_are_made_unique():
    ds = clean_raw_records([raw(record_id="EPIC-0001", text="first"),
                            raw(record_id="EPIC-0001", date="2025-03-01", text="second")], today=TODAY)
    ids = [r.record_id for r in ds.records]
    assert sorted(ids) == ["EPIC-0001", "EPIC-0001-2"]
    renamed = next(r for r in ds.records if r.record_id == "EPIC-0001-2")
    assert any(a.field == "record_id" for a in renamed.audit)


@pytest.mark.parametrize("raw_flag,expected,audited", [
    ("", "N", False), ("H", "H", False), ("high", "H", True), ("HH", "HH", False), ("LL", "LL", False),
    ("C", "AA", True), ("CRIT", "AA", True), ("AA", "AA", False),
    (">", "A", True), ("Q", "A", True),  # unknown flags are never treated as normal
])
def test_lab_flags_keep_criticals_and_never_default_unknown_to_normal(raw_flag, expected, audited):
    code, audit = C.normalize_flag(raw_flag)
    assert code == expected
    assert bool(audit) == audited


def test_critical_flag_survives_into_panel_text():
    ds = clean_raw_records([lab(test="Potassium", value="7.1", flag="HH")], today=TODAY)
    assert ds.records[0].results[0].flag == "HH" and "[CRITICAL HIGH]" in ds.records[0].text


def test_future_four_digit_birth_year_is_dropped_not_shifted():
    d, audit = C.parse_date("2058-03-14", "birth_date", is_birth_date=True, today=TODAY)
    assert d is None and "future" in audit[0].reason


def test_impossible_leap_day_birth_date_does_not_crash():
    assert C.parse_date("2400-02-29", "birth_date", is_birth_date=True, today=TODAY)[0] is None


@pytest.mark.parametrize("value", ["March 2019", "2019", "Mar 3"])
def test_partial_dates_are_rejected_not_completed_from_today(value):
    d, audit = C.parse_date(value, "record_date", today=TODAY)
    assert d is None and "incomplete" in audit[0].reason


def test_bad_files_are_reported_and_skipped(tmp_path):
    import json

    from backend.ingestion.pipeline import run_ingestion

    (tmp_path / "good.json").write_text(json.dumps({"records": [raw()]}))
    (tmp_path / "no_records.json").write_text(json.dumps({"source": "x", "items": [1]}))
    (tmp_path / "mixed.json").write_text(json.dumps({"records": ["oops", raw(record_id="R2", text="ok")]}))
    (tmp_path / "broken.json").write_text("{not json")
    (tmp_path / "._good.json").write_bytes(b"\x00\x05\x16\x07Mac OS X")
    (tmp_path / "folder.json").mkdir()
    ds = run_ingestion(tmp_path, today=TODAY)
    assert {r.record_id for r in ds.records} == {"R1", "R2"}
    reasons = {r.source: r.reason for r in ds.rejected}
    assert set(reasons) == {"no_records.json", "mixed.json#0", "broken.json"}


def test_csv_with_byte_order_mark_keeps_mrn(tmp_path):
    from backend.ingestion.loaders import load_lab_csv

    f = tmp_path / "labs.csv"
    f.write_bytes("﻿MRN,Patient Name,DOB,Sex,Collected,Panel,Test,LOINC,Result,Units,Ref Range,Flag\n"
                  "1001,Maria Gonzalez,1958-03-14,F,2025-02-10,P,T,,1,u,,\n".encode())
    assert load_lab_csv(f)[0]["mrn"] == "1001"


@pytest.mark.parametrize("sep", ["----", "-----", "---- ", "  ===  ", None])
def test_txt_header_found_with_separator_variants(tmp_path, sep):
    f = tmp_path / "note.txt"
    header = "PATIENT: Doe, Jane\nMRN: 77\nDOB: 1990-01-01\nDATE OF SERVICE: 2025-01-02\n"
    f.write_text(header + (f"{sep}\n" if sep else "") + "Body text: stable.")
    ds = clean_raw_records(load_txt(f), today=TODAY)
    assert not ds.rejected
    assert ds.records[0].patient_mrn == "MRN-000077" and ds.records[0].text == "Body text: stable."


@pytest.mark.parametrize("value,expected", [("MARY ANN", "Mary Ann"), ("DE LA CRUZ", "De La Cruz"),
                                            ("McDonald", "McDonald"), ("de la Cruz", "de la Cruz"),
                                            ("O'BRIEN", "O'Brien"), ("smith-jones", "Smith-Jones")])
def test_name_capitalization(value, expected):
    assert C._title(value) == expected


def test_record_ids_are_fhir_safe_and_unique_after_sanitizing():
    """Code review: DOC_2025:17 and DOC-2025-17 became the same FHIR id (pipeline crash), and sanitized ids lost
    their audit trail because the audit table was keyed by the raw id."""
    import re

    from backend.ingestion.pipeline import MAX_RECORD_ID

    fhir_id_rule = re.compile(r"[A-Za-z0-9\-.]{1,64}")  # FHIR R4 id datatype

    recs = [raw(record_id="DOC_2025:17", text="one"), raw(record_id="DOC-2025-17", text="two"),
            raw(record_id="X" * 80, text="three"), raw(record_id="Y" * 80, date="2025-03-01", text="four"),
            raw(record_id="Y" * 80, date="2025-04-01", text="five")]
    ids = [r.record_id for r in clean_raw_records(recs, today=TODAY).records]
    assert len(set(ids)) == len(ids)
    assert all(fhir_id_rule.fullmatch(i) and len(i) <= MAX_RECORD_ID for i in ids)  # already a valid FHIR id
    assert all(fhir_id_rule.fullmatch(i + "-99") for i in ids)  # Observation suffixes fit without truncation
