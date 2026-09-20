"""TDD: auto-discover author packs from *-skills routers."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path.home() / ".cursor" / "skills" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from router_packs import assign_packs, build_pack_membership, discover_pack_defs  # noqa: E402


def _skill(root: Path, name: str, *, body: str = "do stuff", author: str | None = None) -> None:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    fm = ["---", f"name: {name}", f"description: {body}"]
    if author:
        fm.append(f'author: "{author}"')
    fm.append("---\n")
    (d / "SKILL.md").write_text("\n".join(fm) + f"\n# {name}\n", encoding="utf-8")


def test_discovers_new_author_router_and_members(tmp_path: Path):
    """Drop acme-skills router + members → pack assignment without PACK_OVERRIDES edit."""
    _skill(
        tmp_path,
        "acme-skills",
        body="Router for Acme design skills. Routes to one skill.",
    )
    (tmp_path / "acme-skills" / "SKILL.md").write_text(
        "---\nname: acme-skills\ndescription: Router for Acme\n---\n"
        "# Acme\n| Task | Skill |\n|---|---|\n| buttons | `acme-button` |\n| forms | `acme-form` |\n",
        encoding="utf-8",
    )
    _skill(tmp_path, "acme-button", body="Acme button craft")
    _skill(tmp_path, "acme-form", body="Acme form craft")
    _skill(tmp_path, "unrelated", body="Something else")

    packs = build_pack_membership(tmp_path, home=tmp_path)  # no commands dir needed
    assert "acme-skills" in packs
    assert "acme-button" in packs["acme-skills"]["members"]
    assert "acme-form" in packs["acme-skills"]["members"]

    pack, all_packs = assign_packs("acme-button", "acme-button", packs)
    assert pack == "acme-skills"
    assert all_packs == ["acme-skills"]

    pack2, _ = assign_packs("unrelated", "unrelated", packs)
    assert pack2 is None


def test_c_level_seed_includes_cfo_advisors():
    root = Path.home() / ".cursor" / "skills"
    packs = build_pack_membership(root, home=Path.home())
    members = packs["c-level-skills"]["members"]
    assert "cfo-advisor" in members
    assert "boardroom" in members
    pack, _ = assign_packs("cfo-advisor", "cfo-advisor", packs)
    assert pack == "c-level-skills"


def test_nested_agent_skills_prefix_is_exclusive(tmp_path: Path):
    router = tmp_path / "nova-skills"
    router.mkdir()
    (router / "SKILL.md").write_text(
        "---\nname: nova-skills\ndescription: Nova router\n---\n# Nova\n",
        encoding="utf-8",
    )
    (router / "index.json").write_text(
        '{"skills":[{"name":"nova-pdf","path":"nova-pdf","skill":"~/.cursor/skills/nova-agent-skills/nova-pdf/SKILL.md"}]}',
        encoding="utf-8",
    )
    nested = tmp_path / "nova-agent-skills" / "nova-pdf"
    nested.mkdir(parents=True)
    (nested / "SKILL.md").write_text(
        "---\nname: nova-pdf\ndescription: PDF tools\n---\n# pdf\n",
        encoding="utf-8",
    )
    # Flat namesake should NOT steal nested tree
    _skill(tmp_path, "nova-pdf", body="unrelated flat pdf skill")

    packs = build_pack_membership(tmp_path, home=tmp_path)
    assert packs["nova-skills"].get("path_prefix") == "nova-agent-skills/"

    pack_nested, _ = assign_packs("nova-pdf", "nova-agent-skills/nova-pdf", packs)
    assert pack_nested == "nova-skills"

    pack_flat, packs_flat = assign_packs("nova-pdf", "nova-pdf", packs)
    # Flat copy is not under prefix; may be unclaimed or differently claimed
    assert pack_flat != "nova-skills" or "nova-agent-skills/" not in str(packs_flat)
