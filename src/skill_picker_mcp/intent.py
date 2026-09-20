"""Deterministic intent parse + query rewrite for skill routing.

Mirrors the research-skill Architecture C idea (classify domain first) and
Elastic's advice: keep the original query as the must-match, enrich with
synonyms/entities as soft boosters — no free-form LLM rewrite.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Domain signals → catalog primary categories (same family as index._CUE_CATS)
_DOMAIN_SIGNALS: list[tuple[re.Pattern[str], str, tuple[str, ...]]] = [
    (re.compile(r"\b(reticle|playwright|e2e|tdd|verify\s+ui|flaky|xctest|smoke\s+test)\b", re.I), "testing", ("test", "qa", "verify")),
    (re.compile(r"\b(react|next\.?js|vue|svelte|tailwind|frontend|typescript|jsx)\b", re.I), "frontend", ("react", "frontend", "ui")),
    (re.compile(r"\b(figma|impeccable|design\s+taste|visual\s+design|ui/?ux|landing\s+page)\b", re.I), "ui-ux", ("design", "ui", "ux")),
    (re.compile(r"\b(api|postgres|graphql|backend|fastapi|django|apache.?age|ag_catalog|sql)\b", re.I), "backend", ("backend", "database", "api")),
    (re.compile(r"\b(graphrag|rag|vector\s+search|sqlite.?vec|hybrid\s+retriev|embedding|rerank)\b", re.I), "agents-ai", ("rag", "retrieval", "embedding")),
    (re.compile(r"\b(seo|serp|backlink|schema\.org|sitemap)\b", re.I), "seo", ("seo", "search")),
    (re.compile(r"\b(ios|swift|android|flutter|react.?native|speech.?analyzer|foundation.?models|xctest)\b", re.I), "mobile", ("ios", "swift", "mobile")),
    (re.compile(r"\b(docker|kubernetes|k8s|ci/?cd|devops|terraform)\b", re.I), "devops", ("docker", "devops", "infra")),
    (re.compile(r"\b(mcp|fastmcp|skill.?picker|agent\s+skill)\b", re.I), "agents-ai", ("mcp", "agent", "skill")),
]

_ACTION_RE = re.compile(
    r"\b(add|build|fix|debug|refactor|design|implement|create|optimize|migrate|"
    r"test|verify|deploy|research|write|review|dockerize|package)\b",
    re.I,
)

# Synonym / abbreviation enrichment (entity expansion, Elastic-style)
_ENRICH: list[tuple[re.Pattern[str], tuple[str, ...]]] = [
    (re.compile(r"\bAGE\b"), ("Apache AGE", "cypher", "graph", "postgres")),
    (re.compile(r"\bGraphRAG\b", re.I), ("knowledge graph", "retrieval", "rag")),
    (re.compile(r"\bNext\.?js\b", re.I), ("React", "App Router", "frontend")),
    (re.compile(r"\bReticle\b", re.I), ("in-app verification", "act_and_wait", "e2e")),
    (re.compile(r"\bXCTest\b", re.I), ("iOS unit test", "Swift testing")),
    (re.compile(r"\bFTS5?\b", re.I), ("full text search", "BM25", "sqlite")),
    (re.compile(r"\bsqlite-vec\b", re.I), ("vector search", "knn", "embedding")),
    (re.compile(r"\bMCP\b"), ("Model Context Protocol", "stdio", "tools")),
]


@dataclass
class Intent:
    """Parsed task intent for transparent routing."""

    original: str
    action: str | None = None
    categories: list[str] = field(default_factory=list)
    enrich_terms: list[str] = field(default_factory=list)
    # Text used for FTS + dense retrieval (original + enrichments)
    rewrite: str = ""

    def as_dict(self) -> dict:
        return {
            "original": self.original,
            "action": self.action,
            "categories": self.categories,
            "enrich_terms": self.enrich_terms,
            "rewrite": self.rewrite,
        }


def parse_intent(task: str) -> Intent:
    """Classify domains + expand entities without an LLM call."""
    text = (task or "").strip()
    if not text:
        return Intent(original="", rewrite="")

    action_m = _ACTION_RE.search(text)
    action = action_m.group(1).lower() if action_m else None

    categories: list[str] = []
    enrich: list[str] = []
    for rx, cat, boosts in _DOMAIN_SIGNALS:
        if rx.search(text):
            if cat not in categories:
                categories.append(cat)
            for b in boosts:
                if b.lower() not in {e.lower() for e in enrich}:
                    enrich.append(b)

    for rx, terms in _ENRICH:
        if rx.search(text):
            for t in terms:
                if t.lower() not in {e.lower() for e in enrich} and t.lower() not in text.lower():
                    enrich.append(t)

    # Elastic pattern: original must stay; enrichments are boosters appended
    rewrite = text if not enrich else f"{text} {' '.join(enrich)}"
    return Intent(
        original=text,
        action=action,
        categories=categories,
        enrich_terms=enrich,
        rewrite=rewrite,
    )
