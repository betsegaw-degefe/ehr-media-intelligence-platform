"""Semantic search over FHIR document text and AI summaries.

Embeddings: sentence-transformers all-MiniLM-L6-v2 (384-d, runs on CPU, ~10 ms per query).
Vector store: ChromaDB, cosine space, persisted to data/chroma. Metadata on every vector lets Chroma
filter by resource type and date range before ranking.
"""

from __future__ import annotations

import re
import time
from datetime import date
from pathlib import Path
from typing import Any, Protocol

import chromadb

from backend.config import EMBEDDING_MODEL

COLLECTION = "ehr_records"
SUMMARY_TYPE = "AISummary"  # pseudo resource type for patient-level AI summaries
RESOURCE_TYPES = ["DocumentReference", "DiagnosticReport", SUMMARY_TYPE]


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
    flat = " ".join(text.split())
    return flat if len(flat) <= n else flat[:n].rsplit(" ", 1)[0] + "…"


class SearchIndex:
    def __init__(self, embedder: Embedder, persist_dir: Path | None = None, collection: str = COLLECTION):
        self.embedder = embedder
        self.client = (chromadb.PersistentClient(path=str(persist_dir)) if persist_dir
                       else chromadb.EphemeralClient())
        self.collection_name = collection
        self.collection = self.client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"}, embedding_function=None)

    # ---------------------------------------------------------------------------------- build
    def rebuild(self, resources: list[dict[str, Any]], summaries: dict[str, dict[str, Any]]) -> int:
        """resources: rows from store.all_resources(); summaries: patient_id -> StoredSummary dict."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            self.collection_name, metadata={"hnsw:space": "cosine"}, embedding_function=None)

        ids, docs, metas = [], [], []
        latest_date: dict[str, str] = {}
        patient_meta: dict[str, tuple[str, str]] = {}
        for r in resources:
            pid = r["patient_id"]
            patient_meta[pid] = (r["mrn"], r["patient_name"])
            if r["record_date"] and r["record_date"] > latest_date.get(pid, ""):
                latest_date[pid] = r["record_date"]
            s = summaries.get(pid) or {}
            ids.append(f"{r['resource_type']}/{r['resource_id']}")
            docs.append(f"{r['title']}. {r['subtype'] or ''}.\n{r['text']}")
            metas.append({
                "resource_type": r["resource_type"], "resource_id": r["resource_id"], "patient_id": pid,
                "mrn": r["mrn"], "patient_name": r["patient_name"], "title": r["title"] or "",
                "subtype": r["subtype"] or "", "date": r["record_date"] or "", "date_int": _date_int(r["record_date"]),
                "summary_snippet": _snippet(s.get("narrative") or s.get("chief_concern") or "", 200),
            })
        for pid, s in summaries.items():
            if pid not in patient_meta:
                continue
            mrn, name = patient_meta[pid]
            text = (f"Chief concern: {s['chief_concern']}\nDiagnoses: {'; '.join(s['key_diagnoses'])}\n"
                    f"Anomalies: {'; '.join(s['flagged_anomalies'])}\n{s['narrative']}")
            ids.append(f"{SUMMARY_TYPE}/{pid}")
            docs.append(text)
            metas.append({
                "resource_type": SUMMARY_TYPE, "resource_id": pid, "patient_id": pid, "mrn": mrn,
                "patient_name": name, "title": "AI clinical summary", "subtype": s.get("model", ""),
                "date": latest_date.get(pid, ""), "date_int": _date_int(latest_date.get(pid)),
                "summary_snippet": _snippet(s.get("narrative") or "", 200),
            })
        if ids:
            self.collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=self.embedder.encode(docs))
        return len(ids)

    def count(self) -> int:
        return self.collection.count()

    # ---------------------------------------------------------------------------------- query
    @staticmethod
    def _where(resource_type: str | None, date_from: date | None, date_to: date | None) -> dict | None:
        clauses: list[dict[str, Any]] = []
        if resource_type:
            clauses.append({"resource_type": resource_type})
        if date_from:
            clauses.append({"date_int": {"$gte": int(date_from.strftime("%Y%m%d"))}})
        if date_to:
            clauses.append({"date_int": {"$lte": int(date_to.strftime("%Y%m%d"))}})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def search(self, query: str, top_k: int = 5, resource_type: str | None = None,
               date_from: date | None = None, date_to: date | None = None) -> tuple[list[dict[str, Any]], float]:
        t0 = time.perf_counter()
        n = self.collection.count()
        if n == 0:
            return [], 0.0
        res = self.collection.query(
            query_embeddings=self.embedder.encode([query]), n_results=min(top_k, n),
            where=self._where(resource_type, date_from, date_to),
            include=["metadatas", "documents", "distances"],
        )
        results = []
        for rank, (meta, doc, dist) in enumerate(zip(res["metadatas"][0], res["documents"][0],
                                                     res["distances"][0]), start=1):
            body = doc.split("\n", 1)[1] if "\n" in doc else doc
            results.append({
                "rank": rank, "score": round(max(0.0, 1.0 - float(dist)), 4),
                "resource_type": meta["resource_type"], "resource_id": meta["resource_id"],
                "patient_id": meta["patient_id"], "mrn": meta["mrn"], "patient_name": meta["patient_name"],
                "date": meta["date"] or None, "title": meta["title"], "subtype": meta["subtype"] or None,
                "snippet": _snippet(body), "summary_snippet": meta["summary_snippet"] or None,
            })
        return results, (time.perf_counter() - t0) * 1000
