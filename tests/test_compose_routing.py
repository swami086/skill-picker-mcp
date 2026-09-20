"""TDD: skill-picker must route via author pack routers and compose multi-author.

Acceptance (user):
1. Task → pick right author pack(s)
2. Combine multiple authors when needed
3. Always route through the pack router (not a random member as the entrypoint)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Import package under test
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skill_picker_mcp.index import (  # noqa: E402
    SkillHit,
    compose_from_hits,
    order_hits_router_first,
    rank_authors,
)


def _hit(
    name: str,
    *,
    pack: str = "",
    primary: str = "ui-ux",
    score: float = -0.05,
    path: str | None = None,
) -> SkillHit:
    return SkillHit(
        name=name,
        path=path or name,
        primary=primary,
        description=f"desc for {name}",
        score=score,
        skill_md="",
        pack=pack,
    )


class TestRankAuthors:
    def test_ranks_packs_by_best_member_score(self):
        hits = [
            _hit("apple-design", pack="emil-skills", score=-0.08),
            _hit("animate", pack="emil-skills", score=-0.05),
            _hit("pdf", pack="anthropic-skills", primary="content", score=-0.07),
            _hit("swift-expert", pack="", primary="mobile", score=-0.09),
        ]
        authors = rank_authors(hits)
        assert [a["pack"] for a in authors] == ["emil-skills", "anthropic-skills"]
        assert authors[0]["router"] == "emil-skills"
        assert authors[0]["best_skill"] == "apple-design"
        assert authors[1]["best_skill"] == "pdf"

    def test_ignores_unpacked_skills_in_author_list(self):
        hits = [_hit("swift-expert", pack="", primary="mobile", score=-0.09)]
        assert rank_authors(hits) == []


class TestComposeRouterFirst:
    def test_primary_is_pack_router_not_member(self):
        """Entry point must be the author router when pack is known."""
        hits = [
            _hit("apple-design", pack="emil-skills", score=-0.09),
            _hit("emil-skills", pack="emil-skills", score=-0.04),
            _hit("animate", pack="emil-skills", score=-0.06),
        ]
        out = compose_from_hits("iOS SwiftUI sheet animation", hits, top_k=3)
        assert out["primary"]["name"] == "emil-skills"
        assert out["primary"]["pack"] == "emil-skills"
        assert out["primary"]["role"] == "router"
        # Recommended specialist from that pack
        assert out["primary"]["recommended_skill"] == "apple-design"

    def test_multi_author_compose_includes_both_routers(self):
        hits = [
            _hit("apple-design", pack="emil-skills", score=-0.09),
            _hit("emil-skills", pack="emil-skills", score=-0.03),
            _hit("pdf", pack="anthropic-skills", primary="content", score=-0.08),
            _hit("anthropic-skills", pack="anthropic-skills", primary="agents-ai", score=-0.02),
            _hit("swift-expert", pack="", primary="mobile", score=-0.07),
        ]
        out = compose_from_hits(
            "polish SwiftUI sheet animation and export a PDF report",
            hits,
            top_k=4,
        )
        packs = set(out["packs_used"])
        assert "emil-skills" in packs
        assert "anthropic-skills" in packs
        # Supporting authors also enter via routers
        support_names = {s["name"] for s in out["supporting"]}
        assert "anthropic-skills" in support_names
        assert out["authors"][0]["pack"] == "emil-skills"
        assert "anthropic-skills" in {a["pack"] for a in out["authors"]}

    def test_instruction_requires_router_then_member(self):
        hits = [
            _hit("pdf", pack="anthropic-skills", primary="content", score=-0.08),
            _hit("anthropic-skills", pack="anthropic-skills", primary="agents-ai", score=-0.05),
        ]
        out = compose_from_hits("merge PDF files", hits, top_k=2)
        assert "router" in out["instruction"].lower()
        assert out["route"]["step1"].startswith("Load router")


class TestComposePrefersAuthorOverUnpacked:
    def test_unpacked_seed_yields_to_ranked_author_router(self):
        hits = [
            _hit("swift-expert", pack="", primary="mobile", score=-0.09),
            _hit("apple-design", pack="emil-skills", score=-0.08),
            _hit("emil-skills", pack="emil-skills", score=-0.03),
        ]
        out = compose_from_hits("iOS SwiftUI Apple design", hits, top_k=3)
        assert out["primary"]["name"] == "emil-skills"
        assert out["primary"]["role"] == "router"
        assert out["primary"]["recommended_skill"] == "apple-design"


class TestComposeSynthesizesRouterWhenMissingFromHits:
    def test_synthesizes_router_entrypoint_when_only_members_hit(self):
        hits = [
            _hit("write-swift", pack="emil-skills", primary="mobile", score=-0.09),
            _hit("apple-design", pack="emil-skills", score=-0.08),
        ]
        out = compose_from_hits("SwiftUI concurrency UI polish", hits, top_k=2)
        assert out["primary"]["name"] == "emil-skills"
        assert out["primary"]["role"] == "router"
        assert out["primary"]["recommended_skill"] in {"write-swift", "apple-design"}


class TestFindResultsRouterFirst:
    def test_find_ordering_leads_with_author_routers(self):
        """find_helpful_skills results must not lead with a pack member."""
        hits = [
            _hit("apple-design", pack="emil-skills", score=-0.09),
            _hit("pdf", pack="anthropic-skills", primary="content", score=-0.08),
            _hit("swift-expert", pack="", primary="mobile", score=-0.07),
            _hit("emil-skills", pack="emil-skills", score=-0.03),
        ]
        ordered = order_hits_router_first(hits)
        assert ordered[0].name == "emil-skills"
        assert ordered[0].pack == "emil-skills"
        assert ordered[1].name == "anthropic-skills"
        assert ordered[1].pack == "anthropic-skills"
        names = [h.name for h in ordered]
        assert names.index("emil-skills") < names.index("apple-design")
        assert "swift-expert" in names
