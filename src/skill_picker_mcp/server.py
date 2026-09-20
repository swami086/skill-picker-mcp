"""MCP server: find / get / list / compose Cursor skills before starting work."""
from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from . import index

mcp = FastMCP(
    "skill-picker",
    instructions=(
        "Call find_helpful_skills or compose_skills BEFORE loading Cursor Agent Skills. "
        "Then Read the returned skill_md_path and follow that skill."
    ),
)


def _hit_dict(h: index.SkillHit, *, include_body: bool = False) -> dict:
    out = {
        "name": h.name,
        "path": index.host_skill_path(h.path),
        "skill_md_path": f"{index.host_skill_path(h.path)}/SKILL.md",
        "primary": h.primary,
        "pack": h.pack or "",
        "score": round(h.score, 4),
        "description": h.description[:400],
    }
    if include_body and h.skill_md:
        out["skill_md"] = h.skill_md[:20_000]
        if len(h.skill_md) > 20_000:
            out["skill_md_truncated"] = True
    return out


@mcp.tool()
def find_helpful_skills(
    task: str,
    category: str | None = None,
    pack: str | None = None,
    top_k: int = 5,
    mode: str = "hybrid",
) -> str:
    """Search Cursor Agent Skills for a task (FTS5 + sqlite-vec hybrid).

    Call FIRST when unsure which of ~700 skills to load. Then Read skill_md_path.
    Prefer pack routers when hits share an author pack; compose across packs when needed.

    Args:
        task: What the user wants done (natural language).
        category: Optional work-mode filter (ui-ux, frontend, backend, seo, …).
        pack: Optional author/bundle filter (emil-skills, anthropic-skills, marketing-skills, …).
        top_k: How many results (1–25).
        mode: hybrid (default) | fts | vec
    """
    hits = index.search(task, category=category, pack=pack, top_k=top_k, mode=mode)
    if not hits:
        return json.dumps(
            {
                "task": task,
                "category": category,
                "pack": pack,
                "mode": mode,
                "results": [],
                "authors": [],
                "hint": "No matches. Broaden terms, omit category/pack, or run reindex_skills.",
            },
            indent=2,
        )
    authors = index.rank_authors(hits)
    ordered = index.order_hits_router_first(hits)
    return json.dumps(
        {
            "task": task,
            "category": category,
            "pack": pack,
            "mode": mode,
            "authors": authors,
            "results": [_hit_dict(h) for h in ordered],
            "route": {
                "step1": "Pick an author from authors[] and load that router SKILL.md",
                "step2": "Let the router select one member (prefer authors[].best_skill)",
                "step3": "If authors[] has multiple packs, compose_skills / load additional routers",
            },
            "next": (
                "Always enter via the pack router (authors[].router). "
                "Then Read best_skill SKILL.md. Use compose_skills when multiple authors apply."
            ),
        },
        indent=2,
    )


@mcp.tool()
def compose_skills(task: str, top_k: int = 3) -> str:
    """Primary skill + supporting skills across categories AND author packs.

    Use when the job spans domains or multiple skill authors (e.g. Emil motion +
    Anthropic PDF + marketing CRO). Diversifies by category and pack.

    Args:
        task: Full user request.
        top_k: Total skills (primary + supporting).
    """
    return json.dumps(index.compose(task, top_k=top_k), indent=2)


@mcp.tool()
def get_skill(name: str, include_body: bool = True) -> str:
    """Load one skill by exact name.

    Args:
        name: Skill name (e.g. impeccable, senior-frontend, emil-skills).
        include_body: Include SKILL.md text when true.
    """
    hit = index.get_skill(name)
    if not hit:
        return json.dumps({"error": f"skill not found: {name}"})
    return json.dumps(_hit_dict(hit, include_body=include_body), indent=2)


@mcp.tool()
def list_skills(
    category: str | None = None,
    pack: str | None = None,
    limit: int = 40,
) -> str:
    """List indexed skills (optional category and/or pack). Prefer find_helpful_skills for tasks.

    Args:
        category: ui-ux | frontend | backend | mobile | seo | marketing | testing | devops | …
        pack: emil-skills | anthropic-skills | marketing-skills | product-skills | …
        limit: Max rows (1–200).
    """
    rows = index.list_skills(category=category, pack=pack, limit=limit)
    return json.dumps(
        {"category": category, "pack": pack, "count": len(rows), "skills": rows},
        indent=2,
    )


@mcp.tool()
def reindex_skills(refresh_catalog: bool = True) -> str:
    """Rescan ~/.cursor/skills → auto-discover author packs + categories → rebuild
    catalog → FTS5 + sqlite-vec index.

    Call after adding a new skill pack (or rely on sessionStart sync-skill-picker.sh).
    Discovers `*-skills` routers, index.json trees, and shared frontmatter authors.
    Returns added/removed names. OpenMemory JSONL is rewritten on disk; MCP does not
    push to OpenMemory.

    Args:
        refresh_catalog: When true (default), run build-skills-catalog.py first
            (writes to /data in Docker). Set false to re-embed from existing catalog only.

    First run downloads MiniLM (~90MB) and embeds all skills — can take a few minutes.
    """
    return json.dumps(
        index.reindex(
            db_path=index.DEFAULT_DB,
            catalog=index.resolve_catalog_path(),
            refresh_catalog=refresh_catalog,
        ),
        indent=2,
    )


def main() -> None:
    try:
        index.ensure_index(db_path=index.DEFAULT_DB, catalog=index.DEFAULT_CATALOG)
    except Exception as exc:  # noqa: BLE001
        print(f"skill-picker: index warm failed: {exc}", flush=True)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
