"""
database.py
-----------
Handles all persistence for the Modus Research Agent:
  - SQLite: reports, sources, findings tables
  - FAISS: vector index for semantic similarity and contradiction detection
"""

import os
import sqlite3
import logging
import threading
import numpy as np
from pathlib import Path
from contextlib import contextmanager
from typing import Optional

import faiss
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DATABASE_PATH = os.getenv("DATABASE_PATH", "./research.db")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./faiss_index")
EMBEDDING_DIM = 384  # BAAI/bge-small-en-v1.5 (FastEmbed)

@contextmanager
def get_connection():
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    # Wait up to 5s for a competing writer instead of raising 'database is locked' immediately.
    conn.execute("PRAGMA busy_timeout=5000;")
    # SQLite ignores declared REFERENCES unless this is enabled per-connection.
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db() -> None:
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS research_sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                question    TEXT    NOT NULL,
                status      TEXT    NOT NULL DEFAULT 'pending',
                created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
                updated_at  TEXT    NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS sub_queries (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  INTEGER NOT NULL REFERENCES research_sessions(id),
                query       TEXT    NOT NULL,
                created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS sources (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id       INTEGER NOT NULL REFERENCES research_sessions(id),
                url              TEXT    NOT NULL,
                title            TEXT,
                raw_content      TEXT,
                relevance_score  REAL    DEFAULT 0.0,
                created_at       TEXT    NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS findings (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      INTEGER NOT NULL REFERENCES research_sessions(id),
                source_id       INTEGER REFERENCES sources(id),
                fact            TEXT    NOT NULL,
                classification  TEXT,
                confidence      REAL    DEFAULT 1.0,
                faiss_index_id  INTEGER,
                created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS contradictions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      INTEGER NOT NULL REFERENCES research_sessions(id),
                finding_a_id    INTEGER NOT NULL REFERENCES findings(id),
                finding_b_id    INTEGER NOT NULL REFERENCES findings(id),
                explanation     TEXT,
                similarity      REAL,
                created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS reports (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  INTEGER NOT NULL UNIQUE REFERENCES research_sessions(id),
                content     TEXT    NOT NULL,
                created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
            );
        """)
    logger.info("SQLite database initialised at %s", DATABASE_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Write helpers
#
# Each helper opens a short-lived connection via ``get_connection()`` (which commits
# on success / rolls back on error) and uses parameterised SQL exclusively — never
# string interpolation — so the data layer is injection-safe by construction.
# ─────────────────────────────────────────────────────────────────────────────

def _last_id(cur: sqlite3.Cursor) -> int:
    """Return the autoincrement id from the just-executed INSERT (never ``None`` on success)."""
    if cur.lastrowid is None:
        raise RuntimeError("INSERT did not produce a row id")
    return cur.lastrowid


def create_session(question: str) -> int:
    """Create a new research session and return its id."""
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO research_sessions (question, status) VALUES (?, ?)",
            (question, "pending"),
        )
        return _last_id(cur)


def insert_sub_queries(session_id: int, queries: list[str]) -> None:
    """Persist the decomposed sub-queries for a session."""
    with get_connection() as conn:
        conn.executemany(
            "INSERT INTO sub_queries (session_id, query) VALUES (?, ?)",
            [(session_id, q) for q in queries],
        )


def insert_source(
    session_id: int,
    url: str,
    title: str,
    raw_content: str,
    relevance_score: float,
) -> int:
    """Persist a collected web source (relevant or not) and return its id."""
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO sources (session_id, url, title, raw_content, relevance_score) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, url, title, raw_content, relevance_score),
        )
        return _last_id(cur)


def insert_finding(
    session_id: int,
    source_id: int,
    fact: str,
    classification: str,
    confidence: float,
    faiss_index_id: int,
) -> int:
    """Persist an extracted finding (linked to its source and FAISS vector) and return its id."""
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO findings "
            "(session_id, source_id, fact, classification, confidence, faiss_index_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, source_id, fact, classification, confidence, faiss_index_id),
        )
        return _last_id(cur)


def insert_contradiction(
    session_id: int,
    finding_a_id: int,
    finding_b_id: int,
    explanation: str,
    similarity: float,
) -> None:
    """Log a detected contradiction between two findings."""
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO contradictions "
            "(session_id, finding_a_id, finding_b_id, explanation, similarity) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, finding_a_id, finding_b_id, explanation, similarity),
        )


def insert_report(session_id: int, content: str) -> None:
    """
    Persist (or replace) the final report for a session.

    ``reports.session_id`` is UNIQUE, so re-running a session upserts rather than failing —
    this keeps the pipeline idempotent.
    """
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO reports (session_id, content) VALUES (?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET "
            "content = excluded.content, created_at = datetime('now')",
            (session_id, content),
        )


def update_session_status(session_id: int, status: str) -> None:
    """Update a session's lifecycle status (e.g. 'searching', 'completed', 'failed')."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE research_sessions SET status = ?, updated_at = datetime('now') WHERE id = ?",
            (status, session_id),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Read helpers (consumed by the contradiction and synthesizer nodes)
# ─────────────────────────────────────────────────────────────────────────────

def get_finding_by_faiss_id(faiss_id: int) -> Optional[sqlite3.Row]:
    """Reverse-map a FAISS vector id back to its finding row (or ``None`` if unknown)."""
    with get_connection() as conn:
        cur = conn.execute(
            "SELECT id, session_id, source_id, fact, classification, confidence, faiss_index_id "
            "FROM findings WHERE faiss_index_id = ? LIMIT 1",
            (faiss_id,),
        )
        return cur.fetchone()


def get_findings(session_id: int) -> list[sqlite3.Row]:
    """Return a session's findings joined to their source (url/title) for citation building."""
    with get_connection() as conn:
        cur = conn.execute(
            "SELECT f.id, f.fact, f.classification, f.confidence, f.faiss_index_id, "
            "       f.source_id, s.url AS url, s.title AS title "
            "FROM findings f LEFT JOIN sources s ON f.source_id = s.id "
            "WHERE f.session_id = ? ORDER BY f.id",
            (session_id,),
        )
        return cur.fetchall()


def get_contradictions(session_id: int) -> list[sqlite3.Row]:
    """Return contradictions logged for a session."""
    with get_connection() as conn:
        cur = conn.execute(
            "SELECT id, finding_a_id, finding_b_id, explanation, similarity "
            "FROM contradictions WHERE session_id = ? ORDER BY id",
            (session_id,),
        )
        return cur.fetchall()


class FAISSStore:
    _index: faiss.IndexFlatL2
    _index_path: str

    def __init__(self, index_path: str = FAISS_INDEX_PATH, dim: int = EMBEDDING_DIM):
        self._index_path = index_path
        # Guards the read-modify-write in add() (ntotal -> add -> persist) and serializes it
        # against search(). FastAPI dispatches run_research() to a threadpool, so two concurrent
        # requests could otherwise interleave here — both reading the same ntotal and colliding
        # on faiss_index_id, or corrupting the on-disk index by writing it simultaneously.
        self._lock = threading.Lock()
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        if Path(index_path).exists():
            self._index = faiss.read_index(index_path)
            logger.info("FAISS index loaded from %s (%d vectors)", index_path, self._index.ntotal)
        else:
            self._index = faiss.IndexFlatL2(dim)
            logger.info("New FAISS index created (dim=%d)", dim)

    def add(self, embedding: np.ndarray) -> int:
        vec = embedding.reshape(1, -1).astype("float32")
        with self._lock:
            row_id = self._index.ntotal
            self._index.add(vec)
            self._persist()
        return row_id

    def search(self, embedding: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        vec = embedding.reshape(1, -1).astype("float32")
        with self._lock:
            if self._index.ntotal == 0:
                return np.array([]), np.array([])
            k = min(k, self._index.ntotal)
            distances, indices = self._index.search(vec, k)
        return distances[0], indices[0]

    def _persist(self) -> None:
        faiss.write_index(self._index, self._index_path)

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal

faiss_store = FAISSStore()