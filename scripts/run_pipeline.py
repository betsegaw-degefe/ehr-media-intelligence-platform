"""Run the full pipeline: ingest -> clean -> FHIR -> validate -> store -> summarize -> index.

Usage
  python scripts/run_pipeline.py                     # everything; LLM summaries if credentials exist
  python scripts/run_pipeline.py --summarizer rules  # no API calls (deterministic fallback summaries)
  python scripts/run_pipeline.py --skip-index        # stop before embedding
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import config, store  # noqa: E402
from backend.fhir.mapper import build_bundles  # noqa: E402
from backend.fhir.validate import validate_bundle, write_report  # noqa: E402
from backend.ingestion.pipeline import run_ingestion  # noqa: E402
from backend.summarize.cache import SummaryCache  # noqa: E402
from backend.summarize.summarizer import (  # noqa: E402
    PROMPT_VERSION, RULES_MODEL, LLMBackend, RulesBackend, StoredSummary, SummaryBackend, notes_hash,
    summarize_bundle,
)

log = logging.getLogger("pipeline")


LLM_SETUP_HINT = ("Set ANTHROPIC_API_KEY, and EHR_LLM_MODEL to an Anthropic model if you override it, "
                  "or run with --summarizer rules.")


def _llm_unusable(e: Exception) -> bool:
    """Errors that will fail every patient the same way (bad or missing credentials, unknown model), as opposed to
    a temporary error such as a 429 or 5xx that the SDK has already retried."""
    import anthropic

    if isinstance(e, (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.NotFoundError)):
        return True
    return isinstance(e, TypeError) and "authentication" in str(e).lower()  # SDK: no credentials resolved


def _pick_backend(choice: str) -> SummaryBackend:
    """Fail fast: one Models API call (free, no tokens) checks the credentials and that the model exists before any
    patient is summarized. `llm` exits with an explanation; `auto` falls back to rule-based summaries."""
    if choice == "rules":
        return RulesBackend()
    try:
        backend = LLMBackend()
        backend.client.models.retrieve(backend.model)
        return backend
    except Exception as e:  # noqa: BLE001 - any failure here means the LLM can't be used for this run
        if choice == "llm":
            raise SystemExit(f"--summarizer llm: the LLM is not usable ({type(e).__name__}: {e}). {LLM_SETUP_HINT}")
        log.warning("LLM unavailable (%s: %s); using rule-based summaries. %s", type(e).__name__, e, LLM_SETUP_HINT)
        return RulesBackend()


def _summarize_patient(pid: str, bundle: dict, dq: list[str], backend: SummaryBackend,
                       allow_fallback: bool) -> tuple[StoredSummary | None, SummaryBackend]:
    """Summarize one patient. Returns the summary (None if it failed) and the backend to use for the next patient.
    A temporary LLM error falls back to rules for this patient only; an error that will repeat for everyone
    (credentials, unknown model) switches the rest of the run, or stops it with --summarizer llm. A patient whose
    summary fails with every backend is skipped, so one bad record can't stop the run."""
    try:
        return summarize_bundle(pid, bundle, backend, dq), backend
    except Exception as e:  # noqa: BLE001 - one failed patient should not stop the run
        log.error("Summary failed for %s with %s: %s: %s", pid, backend.model, type(e).__name__, e)
        if backend.model == RULES_MODEL:  # already the fallback: nothing else to try
            return None, backend
        unusable = _llm_unusable(e)
        if not allow_fallback:
            if unusable:
                raise SystemExit(f"--summarizer llm: the LLM rejected the request ({type(e).__name__}). {LLM_SETUP_HINT}")
            return None, backend
        next_backend: SummaryBackend = backend
        if unusable:
            log.warning("LLM credentials or model rejected; using rule-based summaries for the remaining patients.")
            next_backend = RulesBackend()
        try:
            return summarize_bundle(pid, bundle, RulesBackend(), dq), next_backend
        except Exception as e2:  # noqa: BLE001
            log.error("Rule-based summary also failed for %s: %s: %s", pid, type(e2).__name__, e2)
            return None, next_backend


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    ap.add_argument("--summarizer", choices=["auto", "llm", "rules"], default="auto")
    ap.add_argument("--skip-summaries", action="store_true")
    ap.add_argument("--skip-index", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "huggingface_hub", "chromadb", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    out = config.OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    # ---- Task 1: ingestion & cleaning ------------------------------------------------
    ds = run_ingestion(args.raw_dir)
    (out / "clean_dataset.json").write_text(ds.model_dump_json(indent=2))
    audit = {"patients": {p.patient_id: [a.model_dump() for a in p.audit] for p in ds.patients},
             "records": {r.record_id: {"source": r.source, "changes": [a.model_dump() for a in r.audit]}
                         for r in ds.records},
             "rejected": [r.model_dump() for r in ds.rejected]}
    (out / "audit_log.json").write_text(json.dumps(audit, indent=2, default=str))
    log.info("Ingestion: %s", ds.stats)

    # ---- Task 2: FHIR mapping, validation, storage ----------------------------------------
    bundles = build_bundles(ds)
    results = {pid: validate_bundle(b) for pid, b in bundles.items()}
    _, md = write_report(results, out)
    (out / "bundles").mkdir(exist_ok=True)
    for old in (out / "bundles").glob("*.json"):  # patients merged away in this run must not leave files behind
        old.unlink()
    with store.connect(config.DB_PATH) as conn:
        store.save_dataset(conn, ds)
        for pid, b in bundles.items():
            store.save_bundle(conn, pid, b, results[pid])
            (out / "bundles" / f"{pid}.json").write_text(json.dumps(b, indent=2))
    n_valid = sum(r["valid"] for r in results.values())
    log.info("FHIR: %d/%d bundles valid -> %s", n_valid, len(results), md)
    for pid, r in results.items():
        for e in r["errors"]:
            log.error("  %s %s %s: %s", pid, e["resource"], e["path"], e["message"])

    # ---- Task 3: AI summaries (cached) ----------------------------------------------------
    summaries: dict[str, dict] = {}
    failed: list[str] = []
    if not args.skip_summaries:
        backend = _pick_backend(args.summarizer)
        with store.connect(config.DB_PATH) as conn:
            cache = SummaryCache(conn)
            hits = 0
            generated_by: Counter = Counter()
            for p in ds.patients:
                b = bundles[p.patient_id]
                h = store.bundle_content_hash(b)
                # Data-quality notes are part of the summary input, so they are part of the cache key too.
                dq = [f"{a.field}: {a.reason} ({a.old} -> {a.new})" for a in p.audit if a.action == "conflict"]
                dq += [f"MRN {m} merged into {p.mrn} (same name and date of birth)" for m in p.alternate_mrns]
                cached = cache.get(p.patient_id, h, notes_hash(dq), backend.model, PROMPT_VERSION)
                if cached:
                    hits += 1
                    s = cached
                else:
                    s, backend = _summarize_patient(p.patient_id, b, dq, backend, args.summarizer == "auto")
                    if s is None:
                        failed.append(p.patient_id)
                        continue
                    cache.put(s)
                    generated_by[s.model] += 1
                summaries[p.patient_id] = s.model_dump()
                if s.quality_warnings:
                    log.warning("  %s quality: %s", p.patient_id, "; ".join(s.quality_warnings))
            by_model = ", ".join(f"{m}: {n}" for m, n in sorted(generated_by.items())) or "none"
            log.info("Summaries: %d generated (%s), %d cache hits, %d failed", sum(generated_by.values()), by_model,
                     hits, len(failed))
        (out / "summaries.json").write_text(json.dumps(summaries, indent=2))

    # The index uses the same summary the patient page shows: SummaryCache.latest() for the current Bundle
    # (current prompt version, LLM preferred over the fallback). This also covers patients without a summary
    # from this run (--skip-summaries, or a failed call), who keep their cached one.
    with store.connect(config.DB_PATH) as conn:
        cache = SummaryCache(conn)
        shown = {p.patient_id: cache.latest(p.patient_id, store.bundle_content_hash(bundles[p.patient_id]))
                 for p in ds.patients}
    summaries = {pid: s.model_dump() for pid, s in shown.items() if s is not None}

    # ---- Task 4: embeddings + vector index ------------------------------------------------
    if not args.skip_index:
        from backend.search.index import SearchIndex, SentenceTransformerEmbedder

        t0 = time.perf_counter()
        index = SearchIndex(SentenceTransformerEmbedder(), persist_dir=config.CHROMA_DIR)
        with store.connect(config.DB_PATH) as conn:
            n = index.rebuild(store.all_resources(conn), summaries, store.list_patients(conn))
        log.info("Indexed %d documents in %.1fs", n, time.perf_counter() - t0)

    if failed:  # outputs are written, but don't let a partial run look like success
        log.error("No new summary for %d patient(s): %s (they keep any cached summary).", len(failed), ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
