# EHR Media Intelligence Platform: Write-up

**Pipeline:** synthetic JSON/CSV/TXT → cleaned Pydantic records with audit log → FHIR R4 Bundles (validated, SQLite) →
LLM summaries (cached) → MiniLM + ChromaDB → `POST /search` → Tailwind UI. 15 patients, 51 records, 15/15 Bundles valid.

## Tradeoffs and why
- **Conservative identity resolution.** Records are linked only on an exact name + DOB match, and typo'd MRNs are
  merged into the MRN from the most authoritative source. Unattributable records are rejected, not guessed.
  Attaching a note to the wrong patient is worse than leaving it unlinked. The cost is that real-world variants
  (nicknames, misspelled names) aren't matched.
- **US date convention.** Ambiguous slash dates are read as MM/DD. That suits a US Epic export but would be wrong for
  DD/MM sources. Every conversion is audited, so it can be traced.
- **fhir.resources R4B models.** v8 ships R4B rather than R4 4.0.1. The resources I use are structurally identical,
  and I added the checks it skips: required value-set bindings (it accepts `gender: "F"`) and referential integrity.
- **ChromaDB over FAISS.** Chroma provides metadata filtering (type, date range) applied *before* ranking, plus
  persistence. FAISS would need a separate filter layer. At 50–5,000 documents, speed isn't the constraint.
- **Records, not patients, as search hits**, plus one AI-summary document per patient. Clinicians land on the
  specific note or report and can open the patient from there.
- **Rule-based fallback summarizer.** The whole platform and its tests run without an API key, at the cost of a
  second code path. Its output is labelled `rule-based-fallback` with confidence `low`.
- **Compiled Tailwind CSS, committed to the repo.** No third-party script runs on a page that shows patient data,
  and reviewers don't need a build step. The cost is regenerating `styles.css` after changing classes (a test
  catches a stale file).

## What I would improve with more time
OCR for real scanned PDFs/images; probabilistic patient matching (Fellegi-Sunter/MPI) with a human review queue;
an official FHIR validator (HL7 Java validator, US Core profiles) and terminology server checks for LOINC/UCUM
codes; hybrid search (BM25 + embeddings) and a clinical embedding model; an LLM-as-judge evaluation set with
clinician-labelled reference summaries; authentication, PHI audit logging and HIPAA controls; and incremental
re-indexing in place of full rebuilds.

## FHIR and clinical concepts
I didn't need to do specific research for this; I drew on my past experience working with FHIR. The main concepts
I applied:
- **Resource choice:** `DocumentReference` (an indexed pointer to a document, e.g. notes and scanned letters) vs
  `DiagnosticReport` (an interpreted result, e.g. imaging impressions and lab panels), with `Observation` per analyte.
- **Bundles:** type `collection` vs `document`; `fullUrl` and how relative references resolve. `DocumentReference.date`
  is an `instant`, so it needs a time and timezone. R4 `Encounter.class` is a single Coding.
- **Codes:** LOINC document types (18842-5 discharge summary, 11506-3 progress note), the v2-0074 `RAD`/`LAB`
  categories, v2-0203 `MR` identifiers with `use: old` for merged MRNs, and v3 observation interpretation (H/L).

## How I validated AI summary quality
1. **Constrained generation.** A JSON-schema structured output covers every required field. The prompt rules forbid
   unstated facts, require exact values, and ask for "Not documented" rather than guesses. The disclaimer is added
   in code.
2. **Automatic checks on every summary.** The word count must be ≤ 200 (one corrective retry, then a deterministic
   trim). Each diagnosis must be traceable to the source text, each number in an anomaly must appear in the source,
   and each cited record id must exist. Failures are stored and shown in the UI as quality warnings.
3. **Tests.** The tests inject a fabricated diagnosis, an invented lab value and an unknown record id, and assert all
   three are caught. Over-limit outputs trigger the retry and trim.
4. **No manual review of LLM output.** I did not have an LLM API key, so the LLM summarizer has not been run on this
   dataset. The summaries in the demo come from the rule-based fallback, and the LLM integration is tested only with
   a mocked API response. Quality assurance for real LLM output therefore rests on points 1 and 2, which run
   automatically on every summary once a key is configured, and on the tests in point 3.
