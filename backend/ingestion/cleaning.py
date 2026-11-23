"""Pure field-level cleaning functions. Each returns (clean_value, [AuditEntry, ...]) so the caller can
attach a per-record audit trail. None of these functions mutate their inputs.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date, datetime

from dateutil import parser as dateparser

from backend.ingestion.models import AuditEntry, Gender

# --------------------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------------------
# Explicit formats are tried first so behaviour is deterministic. Slash dates are read as US
# month/day/year (the convention of a US Epic export); this is a documented assumption.
_DATE_FORMATS = [
    "%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%m-%d-%Y", "%d %b %Y", "%d %B %Y",
    "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%Y%m%d", "%d-%b-%Y", "%Y/%m/%d",
]


def parse_date(raw: str | None, field: str, *, is_birth_date: bool = False,
               today: date | None = None) -> tuple[date | None, list[AuditEntry]]:
    today = today or date.today()
    if raw is None or not str(raw).strip():
        return None, [AuditEntry(field=field, action="missing", reason="no value in source")]
    s = str(raw).strip()
    parsed: date | None = None
    two_digit_year = False
    for fmt in _DATE_FORMATS:
        try:
            parsed = datetime.strptime(s, fmt).date()
            two_digit_year = "%y" in fmt
            break
        except ValueError:
            continue
    if parsed is None:
        try:  # ISO timestamps (with Z / offsets)
            parsed = datetime.fromisoformat(s.replace("Z", "+00:00")).date()
        except ValueError:
            parsed = _parse_complete_date(s)
            if parsed is None:
                return None, [AuditEntry(field=field, action="missing", old=s,
                                         reason="unparseable or incomplete date (needs day, month and year); "
                                                "value dropped")]
    audits: list[AuditEntry] = []
    if is_birth_date and parsed > today:
        # 2-digit years: strptime maps 00-68 to 20xx, so a DOB of "03/14/58" becomes 2058. That is the
        # only case with a safe correction; a future 4-digit year is a data error, so it is dropped.
        if two_digit_year:
            try:
                fixed = parsed.replace(year=parsed.year - 100)
            except ValueError:  # Feb 29 in a year whose century-earlier twin is not a leap year
                fixed = None
            if fixed is not None:
                audits.append(AuditEntry(field=field, action="normalized", old=s, new=fixed.isoformat(),
                                         reason="2-digit year resolved to a future birth date; "
                                                "shifted back 100 years"))
                return fixed, audits
        return None, [AuditEntry(field=field, action="missing", old=s,
                                 reason="birth date is in the future; value dropped")]
    if parsed.isoformat() != s:
        reason = "2-digit year expanded; " if two_digit_year else ""
        audits.append(AuditEntry(field=field, action="normalized", old=s, new=parsed.isoformat(),
                                 reason=reason + "date converted to ISO-8601"))
    if not is_birth_date and parsed > today:
        audits.append(AuditEntry(field=field, action="conflict", new=parsed.isoformat(),
                                 reason="service date is in the future"))
    return parsed, audits


_SENTINELS = (datetime(2000, 1, 1), datetime(2004, 12, 28))  # differ in year, month and day


def _parse_complete_date(s: str) -> date | None:
    """dateutil fills missing parts from a default (normally *today*). Parsing with two different defaults and
    requiring the same answer rejects partial dates like "March 2019" instead of inventing a day."""
    try:
        a, b = (dateparser.parse(s, dayfirst=False, default=d) for d in _SENTINELS)
    except (ValueError, OverflowError):
        return None
    return a.date() if a.date() == b.date() else None


_TEXT_DATE = re.compile(
    r"\b(\d{4}-\d{2}-\d{2}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.? \d{1,2}, \d{4}|"
    r"\d{1,2}/\d{1,2}/\d{4})\b"
)


def date_from_text(text: str) -> date | None:
    """Fallback for documents whose header has no date: take the first date mentioned in the narrative."""
    m = _TEXT_DATE.search(text or "")
    if not m:
        return None
    try:
        return dateparser.parse(m.group(1), dayfirst=False).date()
    except (ValueError, OverflowError):
        return None


# --------------------------------------------------------------------------------------
# Gender
# --------------------------------------------------------------------------------------
_GENDER_MAP: dict[str, Gender] = {
    "m": "male", "male": "male", "man": "male", "1": "male",
    "f": "female", "female": "female", "woman": "female", "2": "female",
    "o": "other", "other": "other", "x": "other", "nonbinary": "other", "non-binary": "other",
    "u": "unknown", "unknown": "unknown", "unk": "unknown", "0": "unknown", "9": "unknown",
}


def normalize_gender(raw: str | None) -> tuple[Gender, list[AuditEntry]]:
    if raw is None or not str(raw).strip():
        return "unknown", [AuditEntry(field="gender", action="missing", new="unknown",
                                      reason="no value in source; defaulted to FHIR 'unknown'")]
    key = str(raw).strip().lower()
    mapped = _GENDER_MAP.get(key)
    if mapped is None:
        return "unknown", [AuditEntry(field="gender", action="normalized", old=str(raw), new="unknown",
                                      reason="unrecognized gender code")]
    if mapped != str(raw):
        return mapped, [AuditEntry(field="gender", action="normalized", old=str(raw), new=mapped,
                                   reason="mapped to FHIR administrative-gender code")]
    return mapped, []


# --------------------------------------------------------------------------------------
# MRN
# --------------------------------------------------------------------------------------
MRN_DIGITS = 6


def normalize_mrn(raw: str | None) -> tuple[str | None, list[AuditEntry]]:
    """Canonical MRN is 'MRN-' + 6 zero-padded digits. Prefixes, spaces and extra padding are stripped."""
    if raw is None or not str(raw).strip():
        return None, [AuditEntry(field="mrn", action="missing", reason="no MRN in source")]
    digits = re.sub(r"\D", "", str(raw))
    if not digits or int(digits) == 0:
        return None, [AuditEntry(field="mrn", action="missing", old=str(raw), reason="MRN contains no digits")]
    n = int(digits)
    if len(str(n)) > MRN_DIGITS:
        return None, [AuditEntry(field="mrn", action="missing", old=str(raw),
                                 reason=f"MRN longer than {MRN_DIGITS} significant digits")]
    canon = f"MRN-{n:0{MRN_DIGITS}d}"
    if canon != raw:
        return canon, [AuditEntry(field="mrn", action="normalized", old=str(raw), new=canon,
                                  reason="MRN converted to canonical MRN-000000 format")]
    return canon, []


# --------------------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------------------
def _title(part: str) -> str:
    """Fix capitalization only for ALL-CAPS or all-lowercase input. Mixed case (McDonald, de la Cruz) is
    assumed to be deliberate and kept. Every space-, hyphen- and apostrophe-separated piece is capitalized."""
    part = " ".join(part.split())
    if part != part.upper() and part != part.lower():
        return part
    return " ".join("-".join("'".join(w[:1].upper() + w[1:].lower() for w in seg.split("'"))
                             for seg in word.split("-")) for word in part.split(" "))


def normalize_name(full: str | None = None, given: str | None = None,
                   family: str | None = None) -> tuple[str | None, str | None, list[AuditEntry]]:
    audits: list[AuditEntry] = []
    if not (given or family) and full:
        s = " ".join(str(full).split())
        if "," in s:  # "LAST, FIRST"
            family, given = [p.strip() for p in s.split(",", 1)]
        else:  # "First Last" -> last token is the family name
            parts = s.split(" ")
            given, family = " ".join(parts[:-1]) or None, parts[-1]
        audits.append(AuditEntry(field="name", action="normalized", old=full,
                                 new=f"{given} | {family}", reason="split full name into given | family"))
    g = _title(given.strip()) if given else None
    f = _title(family.strip()) if family else None
    if (given and g != given.strip()) or (family and f != family.strip()):
        audits.append(AuditEntry(field="name", action="normalized", old=f"{given} | {family}",
                                 new=f"{g} | {f}", reason="normalized capitalization"))
    if not g and not f:
        audits.append(AuditEntry(field="name", action="missing", reason="no patient name in source"))
    return g, f, audits


# --------------------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------------------
def normalize_text(text: str | None) -> tuple[str, list[AuditEntry]]:
    """Strip trailing whitespace on each line and collapse runs of blank lines."""
    if not text or not str(text).strip():
        return "", [AuditEntry(field="text", action="missing", reason="record has no narrative text")]
    lines = [ln.strip() for ln in str(text).strip().splitlines()]
    out: list[str] = []
    for ln in lines:
        if ln == "" and out and out[-1] == "":
            continue
        out.append(ln)
    clean = "\n".join(out)
    if clean != text:
        return clean, [AuditEntry(field="text", action="normalized",
                                  reason="trimmed whitespace and collapsed blank lines")]
    return clean, []


def content_fingerprint(*parts: str | None) -> str:
    """Whitespace- and case-insensitive hash used for duplicate detection."""
    norm = "|".join(re.sub(r"\s+", " ", (p or "")).strip().lower() for p in parts)
    return hashlib.sha256(norm.encode()).hexdigest()[:16]


# Lab abnormal flags -> FHIR v3 ObservationInterpretation codes. Critical flags keep their own codes.
_FLAG_MAP = {
    "N": "N", "NORMAL": "N", "NEG": "N",
    "H": "H", "HIGH": "H",
    "L": "L", "LOW": "L",
    "HH": "HH", "CH": "HH", "CRITICAL HIGH": "HH",
    "LL": "LL", "CL": "LL", "CRITICAL LOW": "LL",
    "A": "A", "ABN": "A", "ABNORMAL": "A", "*": "A",
    "AA": "AA", "C": "AA", "CRIT": "AA", "CRITICAL": "AA", "!": "AA", "PANIC": "AA",
}


def normalize_flag(raw: str | None) -> tuple[str, list[AuditEntry]]:
    """An empty flag means the lab reported no abnormality (N). An unrecognized flag is *not* assumed normal:
    it becomes A (abnormal, needs review) and is audited."""
    s = " ".join((raw or "").upper().split())
    if not s:
        return "N", []
    if s in _FLAG_MAP:
        code = _FLAG_MAP[s]
        return code, ([] if code == s else [AuditEntry(field="flag", action="normalized", old=raw, new=code,
                                                         reason="mapped to HL7 observation interpretation code")])
    return "A", [AuditEntry(field="flag", action="normalized", old=raw, new="A",
                            reason="unrecognized abnormal flag; treated as abnormal for review, never as normal")]
