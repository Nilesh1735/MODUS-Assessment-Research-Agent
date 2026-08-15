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
EMBEDDING_DIM = 384  # sentence-transformers all-MiniLM-L6-v2

@contextmanager
def get_connection():
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
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

class FAISSStore:
    _index: faiss.IndexFlatL2
    _index_path: str

    def __init__(self, index_path: str = FAISS_INDEX_PATH, dim: int = EMBEDDING_DIM):
        self._index_path = index_path
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        if Path(index_path).exists():
            self._index = faiss.read_index(index_path)
            logger.info("FAISS index loaded from %s (%d vectors)", index_path, self._index.ntotal)
        else:
            self._index = faiss.IndexFlatL2(dim)
            logger.info("New FAISS index created (dim=%d)", dim)

    def add(self, embedding: np.ndarray) -> int:
        vec = embedding.reshape(1, -1).astype("float32")
        row_id = self._index.ntotal
        self._index.add(vec)
        self._persist()
        return row_id

    def search(self, embedding: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        if self._index.ntotal == 0:
            return np.array([]), np.array([])
        vec = embedding.reshape(1, -1).astype("float32")
        k = min(k, self._index.ntotal)
        distances, indices = self._index.search(vec, k)
        return distances[0], indices[0]

    def _persist(self) -> None:
        faiss.write_index(self._index, self._index_path)

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal

faiss_store = FAISSStore()