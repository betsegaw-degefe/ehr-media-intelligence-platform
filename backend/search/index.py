"""Semantic search over FHIR document text and AI summaries.

Embeddings: sentence-transformers all-MiniLM-L6-v2 (384-d, runs on CPU, ~10 ms per query).
Vector store: ChromaDB, cosine space, persisted to data/chroma. Metadata on every vector lets Chroma
filter by resource type and date range before ranking.

The pipeline and the API are separate processes sharing one persisted index. Chroma keeps an in-memory copy per
process, so after the pipeline rebuilds the index the API would keep serving stale vectors. To handle that,
`rebuild()` updates the collection in place (it is never deleted, so an open handle stays valid) and then writes a
generation marker; `search()` reopens the client when the marker changes.
"""

from __future__ import annotations

import re
import time
import uuid
from datetime import date
from pathlib import Path
from typing import Any, Protocol

import chromadb
from chromadb.errors import NotFoundError

from backend.config import EMBEDDING_MODEL

COLLECTION = "ehr_records"
SUMMARY_TYPE = "AISummary"  # pseudo resource type for patient-level AI summaries
RESOURCE_TYPES = ("DocumentReference", "DiagnosticReport", SUMMARY_TYPE)  # single source of truth
GENERATION_FILE = "index_generation"


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32,
                                 show_progress_bar=False).tolist()


def _date_int(d: str | None) -> int:
    """'2025-03-18' -> 20250318 (Chroma range filters need numbers); 0 when unknown."""
    return int(d[:10].replace("-", "")) if d and re.match(r"\d{4}-\d{2}-\d{2}", d) else 0


def _snippet(text: str, n: int = 240) -> str:
    flat = " ".join((text or "").split())
    return flat if len(flat) <= n else flat[:n].rsplit(" ", 1)[0] + "…"


def _clean_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Chroma metadata values must be str/int/float/bool; never store None."""
    return {k: ("" if v is None else v) for k, v in meta.items()}


class SearchIndex:
    def __init__(self, embedder: Embedder, persist_dir: Path | None = None, collection: str = COLLECTION):
        self.embedder = embedder
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self.collection_name = collection
        self._generation: str | None = None
        self._open()

    # ---------------------------------------------------------------------------------- client lifecycle
    def _open(self) -> None:
        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        else:
            self.client = chromadb.EphemeralClient()
        self.collection = self.client.get_or_create_collection(
            self.collection_name, metadata={"hnsw:space": "cosine"}, embedding_function=None)
        self._generation = self._read_generation()

    def _read_generation(self) -> str | None:
        if not self.persist_dir:
            return None
        try:
            return (self.persist_dir / GENERATION_FILE).read_text()
        except OSError:
            return None

    def _reopen(self) -> None:
        # Chroma caches one client "system" per path per process; clear it so the reopened client loads the
        # index as another process left it.
        from chromadb.api.client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
        self._open()

    def _refresh_if_rebuilt(self) -> None:
        if self.persist_dir and self._read_generation() != self._generation:
            self._reopen()

    # ---------------------------------------------------------------------------------- build
    def rebuild(self, resources: list[dict[str, Any]], summaries: dict[str, dict[str, Any]],
                patients: list[dict[str, Any]] | None = None) -> int:
        """resources: rows from store.all_resources(); summaries: patient_id -> StoredSummary dict;
        patients: rows from store.list_patients() (so a summary is indexed even if the patient has no records)."""
        ids, docs, metas = [], [], []
        latest_date: dict[str, str] = {}
        patient_meta: dict[str, tuple[str, str]] = {
            p["patient_id"]: (p["mrn"], p.get("name") or p["mrn"]) for p in (patients or [])}
        for r in resources:
            pid = r["patient_id"]
            patient_meta.setdefault(pid, (r["mrn"], r.get("patient_name") or r["mrn"]))
            if r["record_date"] and r["record_date"] > latest_date.get(pid, ""):
                latest_date[pid] = r["record_date"]
        for r in resources:
            pid = r["patient_id"]
            mrn, name = patient_meta[pid]
            title = r["title"] or r["subtype"] or r["resource_type"]
            s = summaries.get(pid) or {}
            ids.append(f"{r['resource_type']}/{r['resource_id']}")
            docs.append(f"{title}. {r['subtype'] or ''}.\n{r['text'] or ''}")
            metas.append(_clean_meta({
                "resource_type": r["resource_type"], "resource_id": r["resource_id"], "patient_id": pid,
                "mrn": mrn, "patient_name": name, "title": title, "subtype": r["subtype"],
                "date": r["record_date"], "date_int": _date_int(r["record_date"]),
                "snippet": _snippet(r["text"]),
                "summary_snippet": _snippet(s.get("narrative") or s.get("chief_concern") or "", 200),
            }))
        for pid, s in summaries.items():
            if pid not in patient_meta:
                continue  # summary for a patient that no longer exists in the store
            mrn, name = patient_meta[pid]
            text = (f"Chief concern: {s['chief_concern']}\nDiagnoses: {'; '.join(s['key_diagnoses'])}\n"
                    f"Anomalies: {'; '.join(s['flagged_anomalies'])}\n{s['narrative']}")
            ids.append(f"{SUMMARY_TYPE}/{pid}")
            docs.append(text)
            metas.append(_clean_meta({
                "resource_type": SUMMARY_TYPE, "resource_id": pid, "patient_id": pid, "mrn": mrn,
                "patient_name": name, "title": "AI clinical summary", "subtype": s.get("model"),
                "date": latest_date.get(pid), "date_int": _date_int(latest_date.get(pid)),
                "snippet": _snippet(text),
                "summary_snippet": _snippet(s.get("narrative") or "", 200),
            }))

        # Update in place: upsert the new set, then delete anything not in it.
        if ids:
            self.collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=self.embedder.encode(docs))
        stale = [i for i in self.collection.get(include=[])["ids"] if i not in set(ids)]
        if stale:
            self.collection.delete(ids=stale)
        if self.persist_dir:
            self._generation = uuid.uuid4().hex
            (self.persist_dir / GENERATION_FILE).write_text(self._generation)
        return len(ids)

    def count(self) -> int:
        self._refresh_if_rebuilt()
        return self.collection.count()

    # ---------------------------------------------------------------------------------- query
    @staticmethod
    def _where(resource_type: str | None, date_from: date | None, date_to: date | None) -> dict | None:
        clauses: list[dict[str, Any]] = []
        if resource_type:
            clauses.append({"resource_type": resource_type})
        if date_from or date_to:
            # Undated records (date_int 0) never match a date filter.
            lower = int(date_from.strftime("%Y%m%d")) if date_from else 1
            clauses.append({"date_int": {"$gte": lower}})
        if date_to:
            clauses.append({"date_int": {"$lte": int(date_to.strftime("%Y%m%d"))}})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    @staticmethod
    def _fallback_snippet(meta: dict[str, Any], doc: str | None) -> str:
        """For an index built before snippets were stored in metadata: derive it from the document text
        (record documents start with a title line; summary documents don't)."""
        doc = doc or ""
        if meta.get("resource_type") != SUMMARY_TYPE and "\n" in doc:
            doc = doc.split("\n", 1)[1]
        return _snippet(doc)

    def search(self, query: str, top_k: int = 5, resource_type: str | None = None,
               date_from: date | None = None, date_to: date | None = None) -> tuple[list[dict[str, Any]], float]:
        t0 = time.perf_counter()
        self._refresh_if_rebuilt()
        kwargs = dict(query_embeddings=self.embedder.encode([query]), n_results=top_k,
                      where=self._where(resource_type, date_from, date_to),
                      include=["metadatas", "documents", "distances"])
        try:
            res = self.collection.query(**kwargs)
        except NotFoundError:  # collection replaced by another process: reopen once and retry
            self._reopen()
            res = self.collection.query(**kwargs)
        results = []
        for rank, (meta, doc, dist) in enumerate(zip(res["metadatas"][0], res["documents"][0],
                                                     res["distances"][0]), start=1):
            results.append({
                "rank": rank, "score": round(max(0.0, 1.0 - float(dist)), 4),
                "resource_type": meta["resource_type"], "resource_id": meta["resource_id"],
                "patient_id": meta["patient_id"], "mrn": meta["mrn"],
                "patient_name": meta["patient_name"] or meta["mrn"],
                "date": meta["date"] or None, "title": meta["title"], "subtype": meta["subtype"] or None,
                "snippet": meta.get("snippet") or self._fallback_snippet(meta, doc),
                "summary_snippet": meta.get("summary_snippet") or None,
            })
        return results, (time.perf_counter() - t0) * 1000
