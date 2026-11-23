"""Run the full pipeline: ingest -> clean -> FHIR -> validate -> store -> summarize -> index.

Usage
  python scripts/run_pipeline.py                     # everything; Claude summaries if credentials exist
  python scripts/run_pipeline.py --summarizer rules  # no API calls (deterministic fallback summaries)
  python scripts/run_pipeline.py --skip-index        # stop before embedding
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import config, store  # noqa: E402
from backend.fhir.mapper import build_bundles  # noqa: E402
from backend.fhir.validate import validate_bundle, write_report  # noqa: E402
from backend.ingestion.pipeline import run_ingestion  # noqa: E402
from backend.summarize.cache import SummaryCache  # noqa: E402
from backend.summarize.summarizer import (  # noqa: E402
    PROMPT_VERSION, ClaudeBackend, RulesBackend, SummaryBackend, summarize_bundle,
)

log = logging.getLogger("pipeline")


def _pick_backend(choice: str) -> SummaryBackend:
    if choice == "rules":
        return RulesBackend()
    try:
        return ClaudeBackend()
    except Exception as e:  # noqa: BLE001 - e.g. SDK cannot resolve credentials
        if choice == "claude":
            raise
        log.warning("Claude unavailable (%s); using rule-based summaries.", e)
        return RulesBackend()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    ap.add_argument("--summarizer", choices=["auto", "claude", "rules"], default="auto")
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
    if not args.skip_summaries:
        backend = _pick_backend(args.summarizer)
        with store.connect(config.DB_PATH) as conn:
            cache = SummaryCache(conn)
            hits = calls = 0
            for p in ds.patients:
                b = bundles[p.patient_id]
                h = store.bundle_content_hash(b)
                cached = cache.get(p.patient_id, h, backend.model, PROMPT_VERSION)
                if cached:
                    hits += 1
                    s = cached
                else:
                    dq = [f"{a.field}: {a.reason} ({a.old} -> {a.new})" for a in p.audit if a.action == "conflict"]
                    dq += [f"MRN {m} merged into {p.mrn} (same name and date of birth)" for m in p.alternate_mrns]
                    try:
                        s = summarize_bundle(p.patient_id, b, backend, dq)
                    except Exception as e:  # noqa: BLE001 - one failed patient should not stop the run
                        log.error("Summary failed for %s: %s", p.patient_id, e)
                        if isinstance(backend, ClaudeBackend) and args.summarizer == "auto":
                            log.warning("Falling back to rule-based summaries for remaining patients.")
                            backend = RulesBackend()
                            s = summarize_bundle(p.patient_id, b, backend, dq)
                        else:
                            continue
                    cache.put(s)
                    calls += 1
                summaries[p.patient_id] = s.model_dump()
                if s.quality_warnings:
                    log.warning("  %s quality: %s", p.patient_id, "; ".join(s.quality_warnings))
            log.info("Summaries (%s): %d generated, %d cache hits", backend.model, calls, hits)
        (out / "summaries.json").write_text(json.dumps(summaries, indent=2))

    # ---- Task 4: embeddings + vector index ------------------------------------------------
    if not args.skip_index:
        from backend.search.index import SearchIndex, SentenceTransformerEmbedder

        t0 = time.perf_counter()
        index = SearchIndex(SentenceTransformerEmbedder(), persist_dir=config.CHROMA_DIR)
        with store.connect(config.DB_PATH) as conn:
            n = index.rebuild(store.all_resources(conn), summaries)
        log.info("Indexed %d documents in %.1fs", n, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
