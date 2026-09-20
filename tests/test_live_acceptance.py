"""Live acceptance checks against the host catalog / indexed DB (optional).

Run: uv run --with pytest pytest tests/test_live_acceptance.py -v
Skip if SKILL_PICKER_SKIP_LIVE=1 or DB/catalog missing.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skill_picker_mcp import index  # noqa: E402

CATALOG = Path.home() / ".cursor" / "skills" / "skills-catalog.json"
# Prefer docker volume DB if present via env; else local mcp data
DB_CANDIDATES = [
    Path(os.environ["SKILL_PICKER_DB"]) if os.environ.get("SKILL_PICKER_DB") else None,
    Path.home() / ".cursor" / "skill-picker-mcp" / "data" / "skills.db",
]
DB = next((p for p in DB_CANDIDATES if p and p.is_file()), None)

pytestmark = pytest.mark.skipif(
    os.environ.get("SKILL_PICKER_SKIP_LIVE") == "1" or not CATALOG.is_file() or DB is None,
    reason="live catalog/DB unavailable",
)


def test_ios_design_task_prefers_emil_router():
    out = index.compose(
        "design UI/UX for native iOS SwiftUI app Apple motion sheets",
        top_k=4,
        db_path=DB,
    )
    assert out["primary"] is not None
    assert out["primary"]["role"] == "router"
    packs = set(out["packs_used"])
    assert "emil-skills" in packs or out["primary"]["pack"] == "emil-skills"
    assert out["primary"]["name"] == out["primary"]["pack"] or out["primary"]["name"].endswith(
        "-skills"
    )


def test_multi_author_pdf_and_motion():
    out = index.compose(
        "polish SwiftUI sheet animation and export a PDF report",
        top_k=4,
        db_path=DB,
    )
    packs = set(out["packs_used"])
    assert "emil-skills" in packs
    assert "anthropic-skills" in packs
    # Entrypoints are routers
    assert out["primary"]["role"] == "router"
    assert any(s.get("role") == "router" and s.get("pack") == "anthropic-skills" for s in out["supporting"])


def test_find_ranks_authors():
    hits = index.search(
        "design UI/UX for native iOS SwiftUI Apple design",
        top_k=10,
        db_path=DB,
    )
    authors = index.rank_authors(hits)
    assert authors, "expected at least one author pack"
    assert authors[0]["router"].endswith("-skills") or authors[0]["pack"] == "emil-skills"
