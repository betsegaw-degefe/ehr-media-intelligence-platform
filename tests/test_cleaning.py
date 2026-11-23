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
