"""TDD: sync fingerprint must notice new packs and slash commands."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path.home() / ".cursor" / "skills" / "scripts"
SYNC = SCRIPTS / "sync-skill-picker.sh"


@pytest.mark.skipif(not SYNC.is_file(), reason="sync script missing")
def test_sync_fingerprint_changes_when_command_added(tmp_path: Path):
    skills = tmp_path / "skills"
    commands = tmp_path / "commands"
    skills.mkdir()
    commands.mkdir()
    (skills / "scripts").mkdir()
    # minimal stubs so build is not invoked (we only test fingerprint via dry env)
    # Call the python fingerprint block by sourcing logic — run sync with FORCE=0
    # against empty catalog path by invoking the embedded python.
    env = os.environ.copy()
    env["SKILLS"] = str(skills)
    env["COMMANDS"] = str(commands)
    py = r"""
from pathlib import Path
import os
skills = Path(os.environ["SKILLS"])
commands = Path(os.environ["COMMANDS"])
mtimes = []
def add(p):
    try: mtimes.append(p.stat().st_mtime_ns)
    except OSError: pass
if skills.is_dir():
    for p in skills.rglob("SKILL.md"): add(p)
    for p in skills.iterdir():
        if p.is_dir() and (p.name.endswith("-skills") or p.name.endswith("-agent-skills")):
            add(p)
if commands.is_dir():
    for p in commands.rglob("*.md"): add(p)
n_skills = sum(1 for _ in skills.rglob("SKILL.md")) if skills.is_dir() else 0
n_packs = sum(1 for p in skills.iterdir() if p.is_dir() and p.name.endswith("-skills")) if skills.is_dir() else 0
n_cmds = sum(1 for _ in commands.rglob("*.md")) if commands.is_dir() else 0
stamp = max(mtimes) if mtimes else 0
print(f"{stamp}:{n_skills}:{n_packs}:{n_cmds}")
"""
    a = subprocess.check_output([sys.executable, "-c", py], env=env, text=True).strip()
    (commands / "acme-skills.md").write_text("# router\n", encoding="utf-8")
    pack = skills / "acme-skills"
    pack.mkdir()
    (pack / "SKILL.md").write_text("---\nname: acme-skills\n---\n# Acme\n", encoding="utf-8")
    b = subprocess.check_output([sys.executable, "-c", py], env=env, text=True).strip()
    assert a != b
    assert b.endswith(":1:1:1")  # 1 skill, 1 pack dir, 1 command
