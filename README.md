# Skill Picker MCP (local + Docker)

Hybrid retrieval over Cursor skills with **intent understanding**:

1. **Intent parse** — deterministic domain/action/entity expand (no LLM)  
2. **Query rewrite** — keep original + enrichment boosters ([Elastic QR pattern](https://www.elastic.co/search-labs/blog/query-rewriting-llm-search-improve))  
3. **FTS5** — lexical BM25  
4. **sqlite-vec** — dense KNN (MiniLM 384-d)  
5. **RRF** — fuse ranked lists  
6. **Cross-encoder rerank** — `ms-marco-MiniLM-L6-v2` scores (query, skill) pairs  

`find_helpful_skills` returns an `intent` object (categories, enrich_terms, rewrite) for transparency.

## Tools

| Tool | Use |
|------|-----|
| `find_helpful_skills` | Ranked match (`mode`: hybrid \| fts \| vec) |
| `compose_skills` | Primary + cross-category supporting (intent categories) |
| `get_skill` | Load one skill by name |
| `list_skills` | Browse (optional category) |
| `reindex_skills` | Rebuild FTS + vectors |

Env: `SKILL_PICKER_RERANK=0` disables cross-encoder.

## Docker (portable)

stdio MCP: client runs `docker run -i --rm …`.

```bash
cd ~/.cursor/skill-picker-mcp
docker build -t skill-picker-mcp:0.1.0 -t skill-picker-mcp:latest .
./scripts/seed-docker-volume.sh   # optional: seed SQLite volume
```

**Cursor `mcp.json`:**

```json
"skill-picker": {
  "command": "docker",
  "args": [
    "run", "-i", "--rm",
    "-v", "/Users/YOU/.cursor/skills:/skills:ro",
    "-v", "skill-picker-data:/data",
    "-e", "SKILL_PICKER_HOST_PREFIX=/Users/YOU/.cursor/skills",
    "-e", "HF_HUB_OFFLINE=1",
    "skill-picker-mcp:0.1.0"
  ]
}
```

## Native (uv)

```bash
uv run --directory ~/.cursor/skill-picker-mcp skill-picker-mcp
```

Refs: [sqlite-vec](https://alexgarcia.xyz/sqlite-vec/python.html) · [cross-encoder](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) · [hybrid+RRF+rerank](https://blog.gopenai.com/hybrid-search-in-rag-dense-sparse-bm25-splade-reciprocal-rank-fusion-and-when-to-use-which-fafe4fd6156e)
