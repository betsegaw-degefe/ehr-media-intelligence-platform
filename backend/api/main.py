"""FastAPI app: semantic search + patient detail endpoints, and the static frontend.

Run:  uvicorn backend.api.main:app --reload   (after `python scripts/run_pipeline.py`)
"""

from __future__ import annotations

import sqlite3
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend import config, store
from backend.search.index import SUMMARY_TYPE, SearchIndex, SentenceTransformerEmbedder
from backend.summarize.cache import SummaryCache

FRONTEND_DIR = config.ROOT / "frontend"


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500, examples=["abnormal chest imaging"])
    top_k: int = Field(default=5, ge=1, le=20)


class SearchResult(BaseModel):
    rank: int
    score: float = Field(description="Cosine similarity, 0-1 (higher is more relevant)")
    resource_type: str
    resource_id: str
    patient_id: str
    mrn: str
    patient_name: str
    date: str | None
    title: str
    subtype: str | None
    snippet: str
    summary_snippet: str | None


class SearchResponse(BaseModel):
    query: str
    filters: dict[str, Any]
    took_ms: float
    results: list[SearchResult]


def create_app(db_path: Path = config.DB_PATH, index: SearchIndex | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load the embedding model once at startup so each query only pays for one forward pass.
        app.state.index = index or SearchIndex(SentenceTransformerEmbedder(), persist_dir=config.CHROMA_DIR)
        app.state.index.embedder.encode(["warm-up"])
        yield

    app = FastAPI(title="EHR Media Intelligence API", version="1.0.0", lifespan=lifespan,
                  description="Semantic search and AI summaries over synthetic FHIR R4 records.")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    def db() -> sqlite3.Connection:
        if not db_path.exists():
            raise HTTPException(503, "Database not found. Run `python scripts/run_pipeline.py` first.")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "indexed_documents": app.state.index.count()}

    @app.post("/search", response_model=SearchResponse)
    def search(
        body: SearchRequest,
        resource_type: Literal["DocumentReference", "DiagnosticReport", "AISummary"] | None = Query(
            None, description="Filter by FHIR resource type (AISummary = patient-level AI summaries)"),
        date_from: date | None = Query(None, description="Inclusive start date, YYYY-MM-DD"),
        date_to: date | None = Query(None, description="Inclusive end date, YYYY-MM-DD"),
    ) -> SearchResponse:
        if date_from and date_to and date_from > date_to:
            raise HTTPException(422, "date_from must be on or before date_to")
        results, took = app.state.index.search(body.query, body.top_k, resource_type, date_from, date_to)
        return SearchResponse(
            query=body.query, took_ms=round(took, 1), results=[SearchResult(**r) for r in results],
            filters={"resource_type": resource_type, "date_from": date_from, "date_to": date_to})

    @app.get("/patients")
    def patients() -> list[dict[str, Any]]:
        conn = db()
        try:
            return store.list_patients(conn)
        finally:
            conn.close()

    @app.get("/patients/{patient_id}")
    def patient_detail(patient_id: str) -> dict[str, Any]:
        conn = db()
        try:
            patient = store.get_patient(conn, patient_id)
            if not patient:
                raise HTTPException(404, f"Patient {patient_id} not found")
            bundle_row = conn.execute("SELECT content_hash, valid, error_count FROM bundles WHERE patient_id=?",
                                      (patient_id,)).fetchone()
            summary = SummaryCache(conn).latest(patient_id, bundle_row["content_hash"] if bundle_row else None)
            resources = store.patient_resources(conn, patient_id)
            for r in resources:
                r["audit"] = store.record_audit(conn, r["resource_id"])
            return {
                "patient": patient,
                "summary": summary.model_dump() if summary else None,
                "bundle": dict(bundle_row) if bundle_row else None,
                "resources": resources,
            }
        finally:
            conn.close()

    @app.get("/patients/{patient_id}/bundle")
    def patient_bundle(patient_id: str) -> dict[str, Any]:
        conn = db()
        try:
            bundle = store.get_bundle(conn, patient_id)
            if not bundle:
                raise HTTPException(404, f"No bundle for {patient_id}")
            return bundle
        finally:
            conn.close()

    @app.get("/resource-types")
    def resource_types() -> list[str]:
        return ["DocumentReference", "DiagnosticReport", SUMMARY_TYPE]

    if FRONTEND_DIR.exists():
        app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    return app


app = create_app()
