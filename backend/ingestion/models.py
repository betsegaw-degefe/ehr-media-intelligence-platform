"""Canonical intermediate representation produced by the ingestion pipeline.

These models sit between the messy raw inputs and FHIR. They are deliberately *simpler* than FHIR:
one flat record type with a `category` that decides which FHIR resource it becomes.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

Gender = Literal["male", "female", "other", "unknown"]  # FHIR administrative-gender value set

# Which FHIR resource a record maps to:
#   document   -> DocumentReference (notes, discharge summaries, scanned letters)
#   imaging    -> DiagnosticReport (category RAD) with the report text as presentedForm
#   lab        -> DiagnosticReport (category LAB) + one Observation per analyte
RecordCategory = Literal["document", "imaging", "lab"]


class AuditEntry(BaseModel):
    """One change (or notable decision) made to a record during cleaning."""

    field: str
    action: Literal["normalized", "filled", "merged", "deduplicated", "conflict", "missing", "rejected"]
    old: str | None = None
    new: str | None = None
    reason: str


class LabResult(BaseModel):
    test: str
    loinc: str | None = None
    value: str
    unit: str | None = None
    reference_range: str | None = None
    flag: Literal["H", "L", "A", "N"] = "N"  # high / low / abnormal / normal


class CleanPatient(BaseModel):
    patient_id: str                      # stable FHIR id, derived from canonical MRN
    mrn: str                             # canonical form: MRN-000000
    given_name: str | None = None
    family_name: str | None = None
    birth_date: date | None = None
    gender: Gender = "unknown"
    alternate_mrns: list[str] = Field(default_factory=list)  # MRNs merged into this patient
    audit: list[AuditEntry] = Field(default_factory=list)

    @property
    def display_name(self) -> str:
        return " ".join(p for p in [self.given_name, self.family_name] if p) or self.mrn


class CleanRecord(BaseModel):
    record_id: str
    patient_mrn: str
    category: RecordCategory
    doc_type: str                        # e.g. "Discharge Summary", "Imaging Report - CT", "Lab Panel"
    title: str
    record_date: date | None = None
    encounter_id: str | None = None
    text: str
    modality: str | None = None          # imaging only
    results: list[LabResult] = Field(default_factory=list)  # lab only
    source: str                          # file + row/line provenance
    audit: list[AuditEntry] = Field(default_factory=list)


class RejectedRecord(BaseModel):
    source: str
    reason: str
    raw: dict


class CleanDataset(BaseModel):
    patients: list[CleanPatient]
    records: list[CleanRecord]
    rejected: list[RejectedRecord] = Field(default_factory=list)
    stats: dict[str, int] = Field(default_factory=dict)
