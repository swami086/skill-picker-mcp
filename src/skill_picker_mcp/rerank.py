"""Cross-encoder rerank — query/doc relevance after hybrid retrieval.

Uses ms-marco-MiniLM-L6-v2 (sentence-transformers CrossEncoder).
Pipeline: retrieve (BM25+vec+RRF) → score (query, skill text) pairs → reorder.
Disable with SKILL_PICKER_RERANK=0.
"""
from __future__ import annotations

import os
import re
from dataclasses import replace
from typing import Any

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
# Single-token skill names that often collide with English words in tasks
_NAME_COLLISIONS = frozenset(
    {
        "coverage",
        "test",
        "tests",
        "review",
        "run",
        "status",
        "init",
        "plan",
        "guide",
        "extract",
        "search",
        "build",
        "deploy",
        "debug",
    }
)
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


def _collision_penalty(query: str, hit: Any) -> float:
    """Demote skills whose name is a common English word appearing in the query."""
    name = str(hit.name).lower().strip()
    pen = 0.0
    if name in _NAME_COLLISIONS:
        tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        if name in tokens:
            pen += 3.0
    blob = f"{hit.name} {hit.description}"
    if re.search(r"\b(react|next\.?js)\b", query, re.I) and re.search(
        r"webflow|wordpress|shopify", blob, re.I
    ):
        pen += 5.0
    return pen


def rerank(query: str, hits: list[Any], *, top_k: int) -> list[Any]:
    """Rerank hits by cross-encoder score (higher = more relevant). Returns top_k."""
    if not hits or not rerank_enabled() or not query.strip():
        return hits[:top_k]
    candidates = hits[: max(top_k * 3, 15)]
    try:
        model = _get_reranker()
        raw = model.predict([(query, _doc_text(h)) for h in candidates])
        scores = [
            float(s) - _collision_penalty(query, h)
            for h, s in zip(candidates, raw, strict=True)
        ]
    except Exception:
        return hits[:top_k]

    ranked = sorted(
        zip(candidates, scores, strict=True),
        key=lambda x: x[1],
        reverse=True,
    )
    return [replace(h, score=-float(sc)) for h, sc in ranked[:top_k]]
