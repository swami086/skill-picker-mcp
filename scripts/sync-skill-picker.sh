#!/usr/bin/env bash
# Sync skill catalog (+ skill-picker reindex) when skills OR slash-commands change.
# Host build is source of truth (resolves symlinks). Docker prefers that catalog.
# Discovers new *-skills routers, nested *-agent-skills trees, and command packs.
set -euo pipefail

SKILLS="${SKILL_PICKER_HOST_SKILLS:-$HOME/.cursor/skills}"
COMMANDS="${SKILL_PICKER_HOST_COMMANDS:-$HOME/.cursor/commands}"
BUILD="$SKILLS/scripts/build-skills-catalog.py"
DATA="${SKILL_PICKER_DATA:-$HOME/.cursor/skill-picker-mcp/data}"
MARKER="$DATA/.last-skills-sync"
IMAGE="${SKILL_PICKER_IMAGE:-skill-picker-mcp:0.1.0}"
mkdir -p "$DATA"

CUR="$(
  SKILLS="$SKILLS" COMMANDS="$COMMANDS" python3 - <<'PY'
from pathlib import Path
import os

skills = Path(os.environ["SKILLS"])
commands = Path(os.environ["COMMANDS"])
mtimes: list[int] = []

def add(p: Path) -> None:
    try:
        mtimes.append(p.stat().st_mtime_ns)
    except OSError:
        pass

# Skill packages + catalogs
if skills.is_dir():
    for p in skills.rglob("SKILL.md"):
        add(p)
    for p in skills.rglob("index.json"):
        add(p)
    # New empty *-skills dirs still flip fingerprint via dir mtime
    for p in skills.iterdir():
        if p.is_dir() and (p.name.endswith("-skills") or p.name.endswith("-agent-skills")):
            add(p)

# Slash commands (pack routers often live here as *-skills.md)
if commands.is_dir():
    for p in commands.rglob("*.md"):
        add(p)

# Catalog builder + pack discovery
for name in (
    "scripts/build-skills-catalog.py",
    "scripts/router_packs.py",
    "scripts/sync-skill-picker.sh",
):
    add(skills / name)

# Skill count + pack-dir count (catches add/remove with identical mtimes)
n_skills = sum(1 for _ in skills.rglob("SKILL.md")) if skills.is_dir() else 0
n_packs = (
    sum(1 for p in skills.iterdir() if p.is_dir() and p.name.endswith("-skills"))
    if skills.is_dir()
    else 0
)
n_cmds = sum(1 for _ in commands.rglob("*.md")) if commands.is_dir() else 0
stamp = max(mtimes) if mtimes else 0
print(f"{stamp}:{n_skills}:{n_packs}:{n_cmds}")
PY
)"
PREV="$(cat "$MARKER" 2>/dev/null || echo 0)"
FORCE="${SKILL_PICKER_FORCE_SYNC:-0}"
DO_DOCKER="${SKILL_PICKER_SYNC_DOCKER:-1}"
BOUNCE="${SKILL_PICKER_BOUNCE_MCP:-1}"

if [[ "$FORCE" != "1" && "$CUR" == "$PREV" && -f "$SKILLS/skills-catalog.json" ]]; then
  echo "skill-sync: up to date ($CUR)"
  exit 0
fi

echo "skill-sync: rebuilding host catalog (fingerprint $PREV → $CUR)…"
python3 "$BUILD" --root "$SKILLS"

if [[ "$DO_DOCKER" == "1" ]] && docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "skill-sync: reindexing skill-picker ($IMAGE)…"
  docker run --rm \
    -v "$SKILLS:/skills:ro" \
    -v skill-picker-data:/data \
    -e SKILL_PICKER_HOST_PREFIX="$SKILLS" \
    -e SKILL_PICKER_CATALOG_OUT=/data \
    -e SKILL_PICKER_CATALOG=/data/skills-catalog.json \
    -e HF_HUB_OFFLINE=1 \
    "$IMAGE" \
    python -c "from skill_picker_mcp.index import reindex; import json; print(json.dumps(reindex(refresh_catalog=True), indent=2)[:2500])"
  if [[ "$BOUNCE" == "1" ]]; then
    # Stale -i MCP sessions keep old binary/DB handles; stop so next connect remounts.
    ids="$(docker ps -q --filter ancestor="$IMAGE" 2>/dev/null || true)"
    if [[ -n "${ids:-}" ]]; then
      echo "skill-sync: bouncing MCP container(s)…"
      # shellcheck disable=SC2086
      docker stop $ids >/dev/null || true
    fi
  fi
else
  echo "skill-sync: skipped docker reindex (build $IMAGE first)"
  # Host-side reindex when local DB exists
  if [[ -f "$DATA/skills.db" ]]; then
    echo "skill-sync: reindexing host DB…"
    (cd "$HOME/.cursor/skill-picker-mcp" && uv run python -c \
      "from skill_picker_mcp.index import reindex; print(reindex(refresh_catalog=False).get('tell',''))" \
      2>/dev/null) || true
  fi
fi

echo "$CUR" > "$MARKER"
echo "skill-sync: done"
