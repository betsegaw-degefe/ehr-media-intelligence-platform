# EHR Media Intelligence Platform: Write-up

**Pipeline:** synthetic JSON/CSV/TXT → cleaned Pydantic records with audit log → FHIR R4 Bundles (validated, SQLite) →
summaries (rule-based in this submission; LLM path built) → MiniLM + ChromaDB → `POST /search` → Tailwind UI. 15 patients, 51 records, 15/15 Bundles valid.

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
- **Records, not patients, as search hits**, plus one summary document per patient. Clinicians land on the
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

## AI summary quality: what was actually validated
I didn't have an LLM API key, so **no AI (LLM) summaries were generated**, and I could not validate real LLM output.
What actually happened:

1. **The summaries in the demo are rule-based.** All 15 come from a deterministic extractor that copies the chief
   complaint, listed diagnoses, flagged lab values and imaging impressions from the records. Each is labelled
   `rule-based-fallback` with confidence `low`.
2. **The automatic checks ran on those 15 summaries, and all passed.** They were 62–124 words (limit 200), had no
   diagnosis, number or record id that couldn't be traced to the source records, and all carry the disclaimer.
3. **The LLM path is built but has only been tested with mocked API responses.** That covers structured JSON
   output, prompt rules against unstated facts, the word-limit retry and trim, the same automatic checks, caching,
   and error handling. Tests inject a fabricated diagnosis, an invented lab value and an unknown record id, and
   confirm the checks catch them.
4. **Not done:** running the LLM on this dataset and reviewing its summaries against the source records. With a
   key, that would be my first step.
