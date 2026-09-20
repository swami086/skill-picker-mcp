"""Exhaustive quality matrix: right author + best recommended member.

Gold cases encode expected pack membership and (when known) recommended skill.
Run: uv run --with pytest pytest tests/test_quality_matrix.py -v
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
DB_CANDIDATES = [
    Path(os.environ["SKILL_PICKER_DB"]) if os.environ.get("SKILL_PICKER_DB") else None,
    Path.home() / ".cursor" / "skill-picker-mcp" / "data" / "skills.db",
]
DB = next((p for p in DB_CANDIDATES if p and p.is_file()), None)

pytestmark = pytest.mark.skipif(
    os.environ.get("SKILL_PICKER_SKIP_LIVE") == "1" or not CATALOG.is_file() or DB is None,
    reason="live catalog/DB unavailable",
)

# (task, required_packs_any_of_primary, required_packs_in_compose, recommended_any_of)
# required_packs_any_of_primary: primary.pack must be one of these (or empty = skip)
# required_packs_in_compose: every listed pack must appear in packs_used
# recommended_any_of: primary.recommended_skill must be one of these if non-empty
CASES = [
    (
        "design UI/UX for native iOS SwiftUI Apple motion sheets and springs",
        {"emil-skills"},
        {"emil-skills"},
        {"apple-design", "animate", "animate-expo", "write-swift", "review-animations"},
    ),
    (
        "write modern Swift concurrency actors SwiftUI for iOS app",
        {"emil-skills"},
        {"emil-skills"},
        {"write-swift", "apple-design"},
    ),
    (
        "merge and export a polished PDF report from markdown",
        {"anthropic-skills"},
        {"anthropic-skills"},
        {"pdf", "docx", "pptx"},
    ),
    (
        "polish SwiftUI sheet animation and export a PDF report",
        {"emil-skills", "anthropic-skills"},
        {"emil-skills", "anthropic-skills"},
        {"apple-design", "animate", "animate-expo", "write-swift", "pdf", "review-animations"},
    ),
    (
        "practice test-driven development with pytest red green refactor",
        {"engineering-skills", "engineering-advanced-skills"},
        set(),
        {"tdd-guide", "skill-tester", "agentic-tdd"},
    ),
    (
        "build an MCP server with tools and resources for Cursor",
        {"anthropic-skills", "engineering-advanced-skills"},
        set(),
        {"mcp-builder", "mcp-server", "agent-designer"},
    ),
    (
        "improve SEO and answer-engine optimization for a marketing site",
        {"marketing-skills"},
        {"marketing-skills"},
        {
            "aeo",
            "seo",
            "seo-audit",
            "site-architecture",
            "schema-markup",
            "programmatic-seo",
            "blog-seo-check",
            "claude-seo-audit",
            "seo-content",
        },
    ),
    (
        "CFO runway analysis unit economics fundraising dilution board deck",
        {"c-level-skills", "finance-skills", "commercial-skills"},
        {"c-level-skills"},
        {
            "cfo-advisor",
            "cfo-review",
            "boardroom",
            "board-deck-builder",
            "board-prep",
            "commercial-forecaster",
            "research-finance",
        },
    ),
    (
        "write Playwright end-to-end tests for a Next.js web app",
        set(),  # pack varies; check recommended + router role
        set(),
        {"playwright", "webapp-testing", "agentic-tdd", "tdd-guide", "accessibility-tester"},
    ),
    (
        "Docker multi-stage build CI/CD GitHub Actions deploy",
        set(),
        set(),
        {"docker", "ci-cd-pipeline-builder", "build-engineer", "devops", "cli-developer"},
    ),
    (
        "Figma design-to-code implement pixel-perfect React component",
        set(),
        set(),
        {
            "figma-design-to-code",
            "frontend-design",
            "senior-frontend",
            "impeccable",
            "web-design-engineer",
        },
    ),
    (
        "security review threat model OWASP for a web API",
        {"engineering-skills", "engineering-advanced-skills", "c-level-skills"},
        set(),
        {
            "ai-security",
            "cloud-security",
            "ad-security-reviewer",
            "ciso-advisor",
            "ciso-review",
            "security-auditor",
            "security-engineer",
            "security-pen-testing",
            "senior-security",
            "senior-secops",
            "skill-security-auditor",
        },
    ),
]


def _compose(task: str, top_k: int = 5) -> dict:
    return index.compose(task, top_k=top_k, db_path=DB)


@pytest.mark.parametrize(
    "task,primary_packs,compose_packs,recommended",
    CASES,
    ids=[c[0][:48] for c in CASES],
)
def test_compose_quality_matrix(task, primary_packs, compose_packs, recommended):
    out = _compose(task)
    assert out["primary"] is not None, task
    assert out["primary"]["role"] == "router" or not out["primary"].get("pack"), (
        f"expected router when packed: {out['primary']}"
    )
    packs = set(out["packs_used"])
    for p in compose_packs:
        assert p in packs, f"{task}: missing pack {p} in {packs}"
    if primary_packs:
        assert out["primary"]["pack"] in primary_packs or out["primary"]["name"] in primary_packs, (
            f"{task}: primary pack {out['primary'].get('pack')!r} not in {primary_packs}; "
            f"recommended={out['primary'].get('recommended_skill')}"
        )
    rec = out["primary"].get("recommended_skill")
    if recommended and out["primary"].get("pack"):
        # Accept recommended on primary OR any supporting router from required packs
        candidates = {rec} if rec else set()
        for s in out["supporting"]:
            if s.get("recommended_skill"):
                candidates.add(s["recommended_skill"])
        if primary_packs:
            # also allow best from authors list
            for a in out.get("authors") or []:
                if a.get("pack") in primary_packs and a.get("best_skill"):
                    candidates.add(a["best_skill"])
        assert candidates & set(recommended), (
            f"{task}: none of {recommended} in {candidates}; "
            f"primary={out['primary']['name']} packs={packs}"
        )


def test_find_leads_with_router_for_emil_task():
    hits = index.search(
        "design UI/UX for native iOS SwiftUI Apple motion",
        top_k=8,
        db_path=DB,
    )
    ordered = index.order_hits_router_first(hits)
    authors = index.rank_authors(hits)
    assert authors and authors[0]["pack"] == "emil-skills"
    assert ordered[0].name == "emil-skills"
    assert ordered[0].pack == "emil-skills"


def test_multi_author_find_surfaces_multiple_routers():
    hits = index.search(
        "SwiftUI animation PDF export report",
        top_k=12,
        db_path=DB,
    )
    authors = index.rank_authors(hits)
    packs = {a["pack"] for a in authors}
    assert "emil-skills" in packs
    assert "anthropic-skills" in packs
    ordered = index.order_hits_router_first(hits)
    lead = [h.name for h in ordered[:3]]
    assert "emil-skills" in lead or "anthropic-skills" in lead


def test_every_compose_author_has_router_name():
    out = _compose(
        "landing page CRO plus SEO plus motion micro-interactions",
        top_k=5,
    )
    for a in out["authors"]:
        assert a["router"].endswith("-skills") or a["router"] == a["pack"]
        assert a["pack"]
