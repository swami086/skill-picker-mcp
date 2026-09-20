#!/usr/bin/env bash
# Seed Docker volume from host index (skip multi-minute re-embed) then print Cursor mcp.json snippet.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST_SKILLS="${SKILL_PICKER_HOST_SKILLS:-$HOME/.cursor/skills}"
HOST_DB="${SKILL_PICKER_HOST_DB:-$ROOT/data/skills.db}"
IMAGE="${SKILL_PICKER_IMAGE:-skill-picker-mcp:0.1.0}"
VOL="${SKILL_PICKER_VOLUME:-skill-picker-data}"

if [[ ! -f "$HOST_DB" ]]; then
  echo "No host DB at $HOST_DB — run: uv run --directory $ROOT python -c 'from skill_picker_mcp.index import reindex; print(reindex())'" >&2
  exit 1
fi

docker volume create "$VOL" >/dev/null
cid="$(docker create --entrypoint true -v "$VOL:/data" "$IMAGE")"
docker cp "$HOST_DB" "$cid:/data/skills.db"
# WAL sidecars if present
[[ -f "${HOST_DB}-wal" ]] && docker cp "${HOST_DB}-wal" "$cid:/data/skills.db-wal" || true
[[ -f "${HOST_DB}-shm" ]] && docker cp "${HOST_DB}-shm" "$cid:/data/skills.db-shm" || true
docker rm "$cid" >/dev/null
echo "Seeded volume $VOL from $HOST_DB"
echo
cat <<EOF
Cursor mcp.json entry:

  "skill-picker": {
    "command": "docker",
    "args": [
      "run", "-i", "--rm",
      "-v", "${HOST_SKILLS}:/skills:ro",
      "-v", "${VOL}:/data",
      "-e", "SKILL_PICKER_HOST_PREFIX=${HOST_SKILLS}",
      "${IMAGE}"
    ]
  }
EOF
