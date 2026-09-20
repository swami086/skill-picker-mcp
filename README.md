# Skill Picker MCP (local + Docker)

Hybrid retrieval over Cursor skills (`~/.cursor/skills`):

1. **FTS5** — lexical BM25  
2. **sqlite-vec** — dense KNN (`float[384]`, MiniLM)  
3. **RRF** — fuse both  

## Tools

| Tool | Use |
|------|-----|
| `find_helpful_skills` | Ranked match (`mode`: hybrid \| fts \| vec); optional `category` + `pack` |
| `compose_skills` | Primary + supporting across **categories and author packs** |
| `get_skill` | Load one skill by name |
| `list_skills` | Browse by `category` and/or `pack` (e.g. `emil-skills`) |
| `reindex_skills` | Rescan skills dir → rebuild catalog → FTS + vectors; returns `added`/`removed` |

**Dual axis:** every skill has `primary` (work mode: ui-ux, mobile, …) and optional `pack`
(author router: emil-skills, anthropic-skills, …). Prefer pack routers; compose across packs
when a task needs multiple authors. Catalog builder: `~/.cursor/skills/scripts/build-skills-catalog.py`
(+ `router_packs.py`).

## Auto-discovery (new packs / skills / slash commands)

Drop a new `*-skills` router (and members) under `~/.cursor/skills/`, and/or a matching
`~/.cursor/commands/*-skills.md`. No manual pack registry edit required.

`scripts/sync-skill-picker.sh` (also at `~/.cursor/skills/scripts/`) fingerprints:

- every `SKILL.md` + `index.json`
- top-level `*-skills` / `*-agent-skills` dirs
- `~/.cursor/commands/**/*.md`
- catalog builder scripts

On change it rebuilds the host catalog (auto-discovers packs), reindexes the Docker volume,
and bounces any long-lived `skill-picker-mcp` container so the next MCP connect sees the new
index. Wired to Cursor `sessionStart` in `~/.cursor/hooks.json`.

```bash
# Force now
SKILL_PICKER_FORCE_SYNC=1 bash ~/.cursor/skills/scripts/sync-skill-picker.sh
# or MCP tool
# reindex_skills({ refresh_catalog: true })
```

## Docker (portable)

stdio MCP pattern: client runs `docker run -i --rm …` and talks over stdin/stdout.

```bash
cd ~/.cursor/skill-picker-mcp
docker build -t skill-picker-mcp:0.1.0 -t skill-picker-mcp:latest .
# Bounce any long-lived MCP container so Cursor picks up the new image
ids=$(docker ps -q --filter ancestor=skill-picker-mcp:0.1.0); [ -n "$ids" ] && docker stop $ids
# Optional: copy existing host index into a named volume (skips first-run embed)
chmod +x scripts/seed-docker-volume.sh && ./scripts/seed-docker-volume.sh
```

After rebuild: toggle the skill-picker MCP (or call `mcp_auth`) so the next `docker run` uses the new image. A stale `-i` session keeps the old binary until it exits.

**Cursor `mcp.json`:**

```json
"skill-picker": {
  "command": "docker",
  "args": [
    "run", "-i", "--rm",
    "-v", "/Users/YOU/.cursor/skills:/skills:ro",
    "-v", "skill-picker-data:/data",
    "-e", "SKILL_PICKER_HOST_PREFIX=/Users/YOU/.cursor/skills",
    "skill-picker-mcp:0.1.0"
  ]
}
```

| Mount / env | Purpose |
|-------------|---------|
| `/skills:ro` | Catalog + SKILL.md corpus |
| `/data` | SQLite FTS + vec index (named volume = portable) |
| `SKILL_PICKER_HOST_PREFIX` | Paths returned to the host agent for `Read` |

Push anywhere: `docker tag skill-picker-mcp:0.1.0 YOUR_REG/skill-picker-mcp:0.1.0 && docker push …`

One-shot reindex inside the volume:

```bash
docker compose --profile tools run --rm reindex
```

## Native (uv)

```json
"skill-picker": {
  "command": "uv",
  "args": ["run", "--directory", "/Users/YOU/.cursor/skill-picker-mcp", "skill-picker-mcp"]
}
```

```bash
# One-shot: rescan packages + reindex (preferred after adding a skill)
uv run --directory ~/.cursor/skill-picker-mcp python -c \
  "from skill_picker_mcp.index import reindex; print(reindex())"

# Catalog only
uv run --directory ~/.cursor/skill-picker-mcp python \
  ~/.cursor/skills/scripts/build-skills-catalog.py
```

## Env

| Variable | Default |
|----------|---------|
| `SKILL_PICKER_SKILLS_ROOT` | `~/.cursor/skills` (Docker: `/skills`) |
| `SKILL_PICKER_CATALOG` | `$SKILLS_ROOT/skills-catalog.json` |
| `SKILL_PICKER_DB` | `…/data/skills.db` (Docker: `/data/skills.db`) |
| `SKILL_PICKER_HOST_PREFIX` | same as skills root |

Refs: [sqlite-vec Python](https://alexgarcia.xyz/sqlite-vec/python.html) · [MCP Docker stdio](https://github.com/mapbox/mcp-server/blob/main/docs/cursor-setup.md)
