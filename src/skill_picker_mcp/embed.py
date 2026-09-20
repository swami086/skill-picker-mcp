"""Lazy local embeddings for skill-picker (sentence-transformers MiniLM)."""
from __future__ import annotations

from typing import Sequence

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384

_model = None


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Encode texts → L2-normalized float32 vectors (dim=384). Lazy-loads model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBED_MODEL)
    vectors = _model.encode(
        list(texts),
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 32,
    )
    return [v.astype("float32").tolist() for v in vectors]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
