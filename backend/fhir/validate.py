"""Validate Bundles against the FHIR R4 schema (fhir.resources) plus referential-integrity checks.

fhir.resources ships the R4B model set (`fhir.resources.R4B`). R4B is a technical-correction release of
R4 (4.3 vs 4.0.1); for Patient, Encounter, DocumentReference, DiagnosticReport, Observation and Bundle the
structures used here are identical, so passing R4B validation means the JSON is valid R4.

Two layers of checks:
  1. schema    every resource and the Bundle itself is parsed by its fhir.resources model
               (cardinality, datatypes, value-set-bound codes such as gender/status, id/instant formats)
  2. integrity every `Type/id` reference inside the Bundle resolves to an entry in the same Bundle,
               every clinical resource has a subject pointing at the Bundle's Patient
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fhir.resources.R4B import get_fhir_model_class
from pydantic import ValidationError

FHIR_VERSION = "R4 (validated with fhir.resources R4B models, FHIR 4.3 / 4.0.1-compatible)"


# fhir.resources checks structure and datatypes but not terminology, so `gender: "F"` would pass.
# These are the *required* (closed) value-set bindings on the elements this project populates.
REQUIRED_BINDINGS: dict[tuple[str, str], set[str]] = {
    ("Patient", "gender"): {"male", "female", "other", "unknown"},
    ("Encounter", "status"): {"planned", "arrived", "triaged", "in-progress", "onleave", "finished",
                              "cancelled", "entered-in-error", "unknown"},
    ("DocumentReference", "status"): {"current", "superseded", "entered-in-error"},
    ("DocumentReference", "docStatus"): {"preliminary", "final", "amended", "entered-in-error"},
    ("DiagnosticReport", "status"): {"registered", "partial", "preliminary", "final", "amended", "corrected",
                                     "appended", "cancelled", "entered-in-error", "unknown"},
    ("Observation", "status"): {"registered", "preliminary", "final", "amended", "corrected", "cancelled",
                                "entered-in-error", "unknown"},
    ("Bundle", "type"): {"document", "message", "transaction", "transaction-response", "batch",
                         "batch-response", "history", "searchset", "collection"},
}


def _binding_errors(resource: dict[str, Any]) -> list[dict[str, str]]:
    rtype = resource.get("resourceType")
    return [{"path": fld, "message": f"'{resource[fld]}' is not in the required value set {sorted(allowed)}"}
            for (rt, fld), allowed in REQUIRED_BINDINGS.items()
            if rt == rtype and fld in resource and resource[fld] not in allowed]


def _schema_errors(resource: dict[str, Any]) -> list[dict[str, str]]:
    rtype = resource.get("resourceType", "?")
    binding = _binding_errors(resource)
    try:
        get_fhir_model_class(rtype).model_validate(resource)
        return binding
    except KeyError:
        return [{"path": "resourceType", "message": f"unknown resourceType '{rtype}'"}]
    except ValidationError as e:
        return binding + [{"path": ".".join(str(p) for p in err["loc"]) or "(root)", "message": err["msg"]}
                          for err in e.errors()]


def _references(obj: Any, path: str = "") -> list[tuple[str, str]]:
    """Every (json-path, reference string) pair inside a resource."""
    out: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else k
            if k == "reference" and isinstance(v, str):
                out.append((p, v))
            else:
                out += _references(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out += _references(v, f"{path}[{i}]")
    return out


def validate_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    entries = bundle.get("entry", [])
    present = {f"{e['resource']['resourceType']}/{e['resource']['id']}" for e in entries if "resource" in e}
    patients = [r for r in present if r.startswith("Patient/")]

    # Bundle-level schema (structure, type, fullUrl) ...
    shell = {k: v for k, v in bundle.items() if k != "entry"}
    shell["entry"] = [{"fullUrl": e.get("fullUrl")} for e in entries]
    errors += [{"resource": f"Bundle/{bundle.get('id')}", **e} for e in _schema_errors(shell)]

    # ... and each resource against its own model.
    for e in entries:
        res = e.get("resource", {})
        key = f"{res.get('resourceType')}/{res.get('id')}"
        errors += [{"resource": key, **err} for err in _schema_errors(res)]
        for path, ref in _references(res):
            if ref not in present:
                errors.append({"resource": key, "path": path, "message": f"unresolved reference '{ref}'"})
        if res.get("resourceType") != "Patient":
            subj = (res.get("subject") or {}).get("reference")
            if subj is None:
                errors.append({"resource": key, "path": "subject", "message": "missing subject reference"})
            elif subj not in patients:
                errors.append({"resource": key, "path": "subject.reference",
                               "message": f"subject '{subj}' is not this bundle's Patient"})
    if len(patients) != 1:
        errors.append({"resource": f"Bundle/{bundle.get('id')}", "path": "entry",
                       "message": f"expected exactly one Patient, found {len(patients)}"})

    counts: dict[str, int] = {}
    for e in entries:
        t = e.get("resource", {}).get("resourceType", "?")
        counts[t] = counts.get(t, 0) + 1
    return {"bundle_id": bundle.get("id"), "valid": not errors, "resource_counts": counts, "errors": errors}


def write_report(results: dict[str, dict[str, Any]], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    n_valid = sum(r["valid"] for r in results.values())
    summary = {"generated_at": generated, "fhir_version": FHIR_VERSION, "bundles": len(results),
               "valid": n_valid, "invalid": len(results) - n_valid,
               "total_errors": sum(len(r["errors"]) for r in results.values()), "results": results}
    json_path = out_dir / "validation_report.json"
    json_path.write_text(json.dumps(summary, indent=2))

    lines = [f"# FHIR Validation Report", "", f"Generated {generated} · {FHIR_VERSION}", "",
             f"**{n_valid}/{len(results)} bundles valid**, {summary['total_errors']} errors.", "",
             "| Patient | Bundle | Valid | Resources | Errors |", "|---|---|---|---|---|"]
    for pid, r in results.items():
        rc = ", ".join(f"{k} {v}" for k, v in sorted(r["resource_counts"].items()))
        lines.append(f"| {pid} | {r['bundle_id']} | {'✅' if r['valid'] else '❌'} | {rc} | {len(r['errors'])} |")
    bad = {pid: r for pid, r in results.items() if r["errors"]}
    if bad:
        lines += ["", "## Errors", ""]
        for pid, r in bad.items():
            lines.append(f"### {pid}")
            lines += [f"- `{e['resource']}` · `{e['path']}`: {e['message']}" for e in r["errors"]]
    md_path = out_dir / "validation_report.md"
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path
