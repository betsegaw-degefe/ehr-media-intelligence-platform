"""Central paths and settings. Everything is overridable via environment variables."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("EHR_DATA_DIR", ROOT / "data"))
RAW_DIR = Path(os.getenv("EHR_RAW_DIR", DATA_DIR / "raw"))
OUTPUT_DIR = Path(os.getenv("EHR_OUTPUT_DIR", DATA_DIR / "output"))
DB_PATH = Path(os.getenv("EHR_DB_PATH", DATA_DIR / "ehr.db"))
CHROMA_DIR = Path(os.getenv("EHR_CHROMA_DIR", DATA_DIR / "chroma"))

EMBEDDING_MODEL = os.getenv("EHR_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CLAUDE_MODEL = os.getenv("EHR_CLAUDE_MODEL", "claude-opus-5-5")
SUMMARY_WORD_LIMIT = 200

AI_DISCLAIMER = (
    "AI-generated summary for information retrieval only. It is not a clinical decision, "
    "diagnosis, or recommendation. Verify all findings against the source records."
)
