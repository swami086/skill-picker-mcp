"""Cross-encoder rerank — query/doc relevance after hybrid retrieval.

Uses ms-marco-MiniLM-L6-v2 (sentence-transformers CrossEncoder).
Pipeline: retrieve (BM25+vec+RRF) → score (query, skill text) pairs → reorder.
Disable with SKILL_PICKER_RERANK=0.
"""
from __future__ import annotations

import os
from dataclasses import replace
from typing import Any

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
_reranker = None


def rerank_enabled() -> bool:
    return os.environ.get("SKILL_PICKER_RERANK", "1").strip() not in {"0", "false", "no"}


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def _doc_text(hit: Any) -> str:
    body = (getattr(hit, "skill_md", None) or hit.description or "")[:1200]
    return f"{hit.name}\n[{hit.primary}]\n{hit.description}\n{body}"


def rerank(query: str, hits: list[Any], *, top_k: int) -> list[Any]:
    """Rerank hits by cross-encoder score (higher = more relevant). Returns top_k."""
    if not hits or not rerank_enabled() or not query.strip():
        return hits[:top_k]
    candidates = hits[: max(top_k * 3, 15)]
    try:
        model = _get_reranker()
        pairs = [(query, _doc_text(h)) for h in candidates]
        scores = model.predict(pairs)
    except Exception:
        return hits[:top_k]

    ranked = sorted(
        zip(candidates, scores, strict=True),
        key=lambda x: float(x[1]),
        reverse=True,
    )
    return [replace(h, score=-float(sc)) for h, sc in ranked[:top_k]]
