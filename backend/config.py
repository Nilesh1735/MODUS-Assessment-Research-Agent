"""
config.py
---------
Central configuration for the Modus Research Agent AI pipeline.

All tunable knobs (model names, search breadth, contradiction thresholds) live here so that
nodes stay declarative and free of magic numbers. Values may be overridden via environment
variables, which keeps the graph layer decoupled from any particular deployment.

Note: SQLite / FAISS filesystem paths are owned by ``backend.database`` (DATABASE_PATH,
FAISS_INDEX_PATH) and are intentionally not duplicated here.
"""

import os

from dotenv import load_dotenv

load_dotenv()


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# ── LLM (Groq) ───────────────────────────────────────────────────────────────
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
# Deterministic outputs are important for a research/audit pipeline.
LLM_TEMPERATURE: float = _get_float("LLM_TEMPERATURE", 0.0)
# Max tokens the LLM may emit per structured call (report can be long).
LLM_MAX_TOKENS: int = _get_int("LLM_MAX_TOKENS", 4096)
# Cap request timeout so a hung provider call cannot stall the whole graph.
LLM_TIMEOUT_SECONDS: float = _get_float("LLM_TIMEOUT_SECONDS", 60.0)

# ── Query generation ─────────────────────────────────────────────────────────
MIN_SUB_QUERIES: int = _get_int("MIN_SUB_QUERIES", 3)
MAX_SUB_QUERIES: int = _get_int("MAX_SUB_QUERIES", 4)

# ── Web search (Tavily) ──────────────────────────────────────────────────────
TAVILY_MAX_RESULTS: int = _get_int("TAVILY_MAX_RESULTS", 4)
TAVILY_SEARCH_DEPTH: str = os.getenv("TAVILY_SEARCH_DEPTH", "advanced")
# Truncate very long page content before sending to the LLM (token / cost control).
MAX_CONTENT_CHARS: int = _get_int("MAX_CONTENT_CHARS", 6000)

# ── Embeddings ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM: int = _get_int("EMBEDDING_DIM", 384)

# ── Contradiction detection ──────────────────────────────────────────────────
# How many FAISS neighbours to inspect per finding.
CONTRADICTION_TOP_K: int = _get_int("CONTRADICTION_TOP_K", 5)
# Cosine-similarity floor (0..1) above which two findings are considered a candidate
# pair worth an LLM conflict check. Derived from FAISS squared-L2 on normalised vectors
# via  sim = 1 - distance / 2.
CONTRADICTION_SIM_THRESHOLD: float = _get_float("CONTRADICTION_SIM_THRESHOLD", 0.6)
