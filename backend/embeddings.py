import logging
from functools import lru_cache
import numpy as np
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from backend import config

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_embedder():
    """
    Returns a shared FastEmbedEmbeddings instance.
    Uses ONNX runtime instead of PyTorch to fit within Render's 512MB free tier RAM limit.
    """
    logger.info("Loading FastEmbed embedding model (%s)...", config.EMBEDDING_MODEL)
    return FastEmbedEmbeddings(model_name=config.EMBEDDING_MODEL)

def embed(text: str) -> np.ndarray:
    """
    Encode text into a normalised float32 vector of shape (EMBEDDING_DIM,).
    """
    if not text or not text.strip():
        return np.zeros(config.EMBEDDING_DIM, dtype="float32")

    # FastEmbed returns a list of floats. We convert it to a numpy array.
    vector = get_embedder().embed_query(text)
    vector = np.asarray(vector, dtype="float32")
    
    # L2 normalize the vector so FAISS IndexFlatL2 works correctly
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
        
    return vector.reshape(-1)