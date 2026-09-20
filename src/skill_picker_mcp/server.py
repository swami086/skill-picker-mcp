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
    top_k: int = 5,
    mode: str = "hybrid",
) -> str:
    """Search Cursor Agent Skills for a task (FTS5 + sqlite-vec hybrid).

    Call FIRST when unsure which of ~700 skills to load. Then Read skill_md_path.

    Args:
        task: What the user wants done (natural language).
        category: Optional primary filter (ui-ux, frontend, backend, seo, …).
        top_k: How many results (1–25).
        mode: hybrid (default) | fts | vec
    """
    hits = index.search(task, category=category, top_k=top_k, mode=mode)
    if not hits:
        return json.dumps(
            {
                "task": task,
                "category": category,
                "mode": mode,
                "results": [],
                "hint": "No matches. Broaden terms, omit category, or run reindex_skills.",
            },
            indent=2,
        )
    return json.dumps(
        {
            "task": task,
            "category": category,
            "mode": mode,
            "results": [_hit_dict(h) for h in hits],
            "next": "Read skill_md_path for the top result and follow that skill.",
        },
        indent=2,
    )


@mcp.tool()
def compose_skills(task: str, top_k: int = 3) -> str:
    """Primary skill + supporting skills across categories for one task.

    Use when the job spans domains (e.g. design + frontend + verify).

    Args:
        task: Full user request.
        top_k: Total skills (primary + supporting).
    """
    return json.dumps(index.compose(task, top_k=top_k), indent=2)


@mcp.tool()
def get_skill(name: str, include_body: bool = True) -> str:
    """Load one skill by exact name.

    Args:
        name: Skill name (e.g. impeccable, senior-frontend).
        include_body: Include SKILL.md text when true.
    """
    hit = index.get_skill(name)
    if not hit:
        return json.dumps({"error": f"skill not found: {name}"})
    return json.dumps(_hit_dict(hit, include_body=include_body), indent=2)


@mcp.tool()
def list_skills(category: str | None = None, limit: int = 40) -> str:
    """List indexed skills (optional category). Prefer find_helpful_skills for tasks.

    Args:
        category: ui-ux | frontend | backend | mobile | seo | marketing | testing | devops | …
        limit: Max rows (1–200).
    """
    rows = index.list_skills(category=category, limit=limit)
    return json.dumps({"category": category, "count": len(rows), "skills": rows}, indent=2)


@mcp.tool()
def reindex_skills(refresh_catalog: bool = True) -> str:
    """Rescan ~/.cursor/skills → rebuild skills-catalog.json → FTS5 + sqlite-vec index.

    Use after adding/removing a global skill package. Returns added/removed names so the
    agent can tell the user what changed. OpenMemory is NOT updated by this tool
    (catalog builder only rewrites skills-catalog.openmemory.jsonl on disk).

    Args:
        refresh_catalog: When true (default), run build-skills-catalog.py first.
            Set false to re-embed from the existing catalog only.

    First run downloads MiniLM (~90MB) and embeds all skills — can take a few minutes.
    """
    return json.dumps(
        index.reindex(
            db_path=index.DEFAULT_DB,
            catalog=index.DEFAULT_CATALOG,
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
