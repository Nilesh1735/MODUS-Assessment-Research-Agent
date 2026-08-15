"""
embeddings.py
-------------
Thin, lazily-initialised wrapper around the ``sentence-transformers`` embedding model
(``all-MiniLM-L6-v2``, 384 dims) used for semantic similarity and contradiction detection.

Design notes:
  * The model is loaded once, on first use, and reused — loading is expensive (~100 MB) and
    must not happen at import time (that would make ``import backend.graph`` slow and couple
    the graph to model availability).
  * Embeddings are L2-normalised so that FAISS ``IndexFlatL2`` squared distances map cleanly
    to cosine similarity:  cosine = 1 - (squared_L2 / 2).
"""

import logging
import threading
from functools import lru_cache

import numpy as np

from backend import config

logger = logging.getLogger(__name__)

_model_lock = threading.Lock()


@lru_cache(maxsize=1)
def get_embedder():
    """Return the process-wide SentenceTransformer, importing/loading it on first call."""
    # Import inside the function so the (heavy) dependency is only pulled in when embeddings
    # are actually needed, keeping module import side-effect-free.
    from sentence_transformers import SentenceTransformer

    with _model_lock:
        logger.info("Loading embedding model '%s'...", config.EMBEDDING_MODEL)
        model = SentenceTransformer(config.EMBEDDING_MODEL)
        logger.info("Embedding model loaded.")
        return model


def embed(text: str) -> np.ndarray:
    """
    Encode ``text`` into a normalised float32 vector of shape ``(EMBEDDING_DIM,)``.

    The returned array is directly compatible with ``faiss_store.add`` / ``faiss_store.search``.
    """
    if not text or not text.strip():
        # Return a zero vector rather than raising, so an empty finding never crashes the graph.
        return np.zeros(config.EMBEDDING_DIM, dtype="float32")

    vector = get_embedder().encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(vector, dtype="float32").reshape(-1)
