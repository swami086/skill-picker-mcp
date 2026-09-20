"""Hybrid skill index: FTS5 (lexical) + sqlite-vec (dense) with RRF fusion.

sqlite-vec load pattern from https://alexgarcia.xyz/sqlite-vec/python.html
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from sqlite_vec import serialize_float32

from .embed import EMBED_DIM, EMBED_MODEL, embed_query, embed_texts
from .intent import Intent, parse_intent
from .rerank import rerank as cross_encoder_rerank


def _env_path(key: str, default: Path) -> Path:
    raw = os.environ.get(key, "").strip()
    return Path(raw).expanduser() if raw else default


DEFAULT_SKILLS_ROOT = _env_path(
    "SKILL_PICKER_SKILLS_ROOT",
    Path.home() / ".cursor" / "skills",
)
DEFAULT_CATALOG = _env_path(
    "SKILL_PICKER_CATALOG",
    DEFAULT_SKILLS_ROOT / "skills-catalog.json",
)
DEFAULT_DB = _env_path(
    "SKILL_PICKER_DB",
    Path.home() / ".cursor" / "skill-picker-mcp" / "data" / "skills.db",
)
# Paths returned to the host agent (Cursor Read). Inside Docker set to host mount.
HOST_SKILLS_PREFIX = os.environ.get(
    "SKILL_PICKER_HOST_PREFIX",
    str(DEFAULT_SKILLS_ROOT),
).rstrip("/")

BODY_CHARS = 12_000
EMBED_CHARS = 2_500
FRONTMATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n?", re.S)
RRF_K = 60


def host_skill_path(rel: str) -> str:
    """Absolute/tilde path the host agent can Read."""
    rel = rel.strip().lstrip("/")
    return f"{HOST_SKILLS_PREFIX}/{rel}" if rel else HOST_SKILLS_PREFIX


@dataclass
class SkillHit:
    name: str
    path: str
    primary: str
    description: str
    score: float
    skill_md: str


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        conn.enable_load_extension(True)
        import sqlite_vec

        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("SELECT vec_version()").fetchone()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"sqlite-vec failed to load ({exc}). "
            "Use uv-managed Python (this project) — macOS system Python often blocks extensions."
        ) from exc
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS skills (
          name TEXT PRIMARY KEY,
          path TEXT NOT NULL,
          primary_cat TEXT NOT NULL,
          description TEXT NOT NULL,
          tags TEXT NOT NULL,
          skill_md TEXT NOT NULL,
          body TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS skills_fts USING fts5(
          name,
          primary_cat,
          description,
          tags,
          body,
          content='skills',
          content_rowid='rowid'
        );
        """
    )
    conn.commit()


def _ensure_vec_table(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='skills_vec'"
    ).fetchone()
    if row:
        return
    conn.execute(
        f"CREATE VIRTUAL TABLE skills_vec USING vec0(embedding float[{EMBED_DIM}])"
    )
    conn.commit()


def _read_skill_md(skills_root: Path, rel_path: str) -> str:
    p = skills_root / rel_path / "SKILL.md"
    if not p.is_file():
        p = skills_root / rel_path
        if p.is_dir():
            p = p / "SKILL.md"
        elif not str(rel_path).endswith("SKILL.md"):
            p = skills_root / f"{rel_path}/SKILL.md"
    try:
        return p.read_text(encoding="utf-8", errors="replace") if p.is_file() else ""
    except OSError:
        return ""


def _body_from_md(text: str) -> str:
    body = FRONTMATTER_RE.sub("", text, count=1)
    return " ".join(body.split())[:BODY_CHARS]


def _embed_doc(name: str, primary: str, desc: str, body: str) -> str:
    return f"{name}\n[{primary}]\n{desc}\n{body[:EMBED_CHARS]}"


def catalog_mtime(catalog: Path) -> str:
    try:
        return str(catalog.stat().st_mtime_ns)
    except OSError:
        return "0"


def catalog_skill_names(catalog: Path) -> set[str]:
    if not catalog.is_file():
        return set()
    doc = json.loads(catalog.read_text(encoding="utf-8"))
    names: set[str] = set()
    for s in doc.get("skills") or []:
        name = str(s.get("name") or "").strip()
        if name:
            names.add(name)
    return names


def indexed_skill_names(db_path: Path) -> set[str]:
    if not db_path.is_file():
        return set()
    conn = _connect(db_path)
    try:
        _ensure_schema(conn)
        rows = conn.execute("SELECT name FROM skills").fetchall()
        return {r["name"] for r in rows}
    except sqlite3.Error:
        return set()
    finally:
        conn.close()


def rebuild_catalog(*, skills_root: Path = DEFAULT_SKILLS_ROOT) -> dict:
    """Rescan ~/.cursor/skills packages → rewrite skills-catalog.json (+ openmemory jsonl)."""
    script = skills_root / "scripts" / "build-skills-catalog.py"
    if not script.is_file():
        return {"ok": False, "error": f"missing catalog builder: {script}"}
    proc = subprocess.run(
        [sys.executable, str(script), "--root", str(skills_root)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    out = {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "script": str(script),
        "stdout_tail": (proc.stdout or "")[-800:],
    }
    if proc.returncode != 0:
        out["stderr_tail"] = (proc.stderr or "")[-800:]
    else:
        out["total"] = len(catalog_skill_names(skills_root / "skills-catalog.json"))
        out["openmemory_jsonl"] = str(skills_root / "skills-catalog.openmemory.jsonl")
    return out


def needs_reindex(conn: sqlite3.Connection, catalog: Path) -> bool:
    """True when catalog file changed (triggers full rebuild including vectors)."""
    row = conn.execute("SELECT value FROM meta WHERE key='catalog_mtime'").fetchone()
    if not row:
        return True
    return row["value"] != catalog_mtime(catalog)


def has_vectors(conn: sqlite3.Connection) -> bool:
    model = conn.execute("SELECT value FROM meta WHERE key='embed_model'").fetchone()
    if not model or model["value"] != EMBED_MODEL:
        return False
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='skills_vec'"
    ).fetchone()
    if not row:
        return False
    n = conn.execute("SELECT count(*) AS c FROM skills_vec").fetchone()["c"]
    return int(n) > 0


def reindex(
    db_path: Path = DEFAULT_DB,
    catalog: Path = DEFAULT_CATALOG,
    skills_root: Path | None = None,
    *,
    refresh_catalog: bool = True,
) -> dict:
    # Prefer env/default root — catalog JSON "root" may be a host path from another machine.
    skills_root = skills_root or DEFAULT_SKILLS_ROOT
    before = indexed_skill_names(db_path)
    catalog_info: dict | None = None
    if refresh_catalog:
        catalog_info = rebuild_catalog(skills_root=skills_root)
        if not catalog_info.get("ok"):
            return {
                "ok": False,
                "error": "catalog rebuild failed",
                "catalog_rebuild": catalog_info,
                "db": str(db_path),
                "catalog": str(catalog),
            }
        catalog = skills_root / "skills-catalog.json"

    doc = json.loads(catalog.read_text(encoding="utf-8"))
    skills = doc.get("skills") or []

    conn = _connect(db_path)
    _ensure_schema(conn)
    conn.execute("DELETE FROM skills")
    conn.execute("DROP TABLE IF EXISTS skills_fts")
    conn.execute("DROP TABLE IF EXISTS skills_vec")
    conn.execute(
        """
        CREATE VIRTUAL TABLE skills_fts USING fts5(
          name, primary_cat, description, tags, body,
          content='skills', content_rowid='rowid'
        )
        """
    )
    conn.execute(
        f"CREATE VIRTUAL TABLE skills_vec USING vec0(embedding float[{EMBED_DIM}])"
    )

    rows_out: list[tuple] = []
    embed_inputs: list[str] = []
    seen_names: set[str] = set()
    for s in skills:
        name = str(s.get("name") or "").strip()
        rel = str(s.get("path") or name).strip()
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        primary = str(s.get("primary") or "general")
        desc = str(s.get("description") or "")
        tags = ",".join(s.get("tags") or [])
        md = _read_skill_md(skills_root, rel)
        body = _body_from_md(md) if md else desc
        rows_out.append((name, rel, primary, desc, tags, md, body))
        embed_inputs.append(_embed_doc(name, primary, desc, body))

    for row in rows_out:
        conn.execute(
            """
            INSERT INTO skills
              (name, path, primary_cat, description, tags, skill_md, body)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            row,
        )
    conn.execute(
        """
        INSERT INTO skills_fts(rowid, name, primary_cat, description, tags, body)
        SELECT rowid, name, primary_cat, description, tags, body FROM skills
        """
    )

    # Batch embed (first run downloads MiniLM)
    vectors = embed_texts(embed_inputs) if embed_inputs else []
    name_to_rowid = {
        r["name"]: r["rowid"]
        for r in conn.execute("SELECT rowid, name FROM skills").fetchall()
    }
    with conn:
        for (name, *_rest), vec in zip(rows_out, vectors, strict=True):
            rid = name_to_rowid[name]
            conn.execute(
                "INSERT INTO skills_vec(rowid, embedding) VALUES (?, ?)",
                [rid, serialize_float32(vec)],
            )

    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES ('catalog_mtime', ?)",
        (catalog_mtime(catalog),),
    )
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES ('count', ?)",
        (str(len(rows_out)),),
    )
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES ('embed_model', ?)",
        (EMBED_MODEL,),
    )
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES ('embed_dim', ?)",
        (str(EMBED_DIM),),
    )
    conn.commit()
    conn.close()
    after = {name for name, *_ in rows_out}
    added = sorted(after - before)
    removed = sorted(before - after)
    result = {
        "ok": True,
        "indexed": len(rows_out),
        "embedded": len(vectors),
        "embed_model": EMBED_MODEL,
        "embed_dim": EMBED_DIM,
        "db": str(db_path),
        "catalog": str(catalog),
        "mode": "fts5+sqlite-vec",
        "added_count": len(added),
        "removed_count": len(removed),
        "added": added[:50],
        "removed": removed[:50],
        "tell": (
            f"Catalog refreshed: +{len(added)} / -{len(removed)} skills "
            f"(index now {len(rows_out)})."
            if refresh_catalog
            else f"Index rebuilt: {len(rows_out)} skills (+{len(added)} / -{len(removed)})."
        ),
    }
    if catalog_info is not None:
        result["catalog_rebuild"] = catalog_info
        if catalog_info.get("openmemory_jsonl"):
            result["openmemory_note"] = (
                "skills-catalog.openmemory.jsonl rewritten by catalog builder. "
                "Skill Picker does not call OpenMemory; sync batches separately if desired."
            )
    return result


def ensure_index(
    db_path: Path = DEFAULT_DB,
    catalog: Path = DEFAULT_CATALOG,
) -> None:
    conn = _connect(db_path)
    _ensure_schema(conn)
    stale = needs_reindex(conn, catalog)
    conn.close()
    if stale or not db_path.exists():
        reindex(db_path=db_path, catalog=catalog)


def _fts_query(raw: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_./+-]{1,40}", raw)
    tokens = [t for t in tokens if t.lower() not in {"the", "and", "for", "with", "this", "that", "from"}]
    if not tokens:
        return raw.replace('"', " ").strip() or "skill"
    parts = []
    for t in tokens[:24]:
        if re.search(r"[^A-Za-z0-9]", t):
            parts.append(f'"{t}"')
        else:
            parts.append(f"{t}*")
    return " OR ".join(parts)


def _row_to_hit(r: sqlite3.Row, score: float) -> SkillHit:
    return SkillHit(
        name=r["name"],
        path=r["path"],
        primary=r["primary_cat"],
        description=r["description"],
        score=score,
        skill_md=r["skill_md"] or "",
    )


def _fts_search(
    conn: sqlite3.Connection,
    task: str,
    *,
    category: str | None,
    limit: int,
) -> list[SkillHit]:
    q = _fts_query(task)
    params: list = [q]
    where = "skills_fts MATCH ?"
    if category:
        where += " AND s.primary_cat = ?"
        params.append(category)
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT s.name, s.path, s.primary_cat, s.description, s.skill_md,
               bm25(skills_fts, 12.0, 6.0, 8.0, 3.0, 1.0) AS score
        FROM skills_fts
        JOIN skills s ON s.rowid = skills_fts.rowid
        WHERE {where}
        ORDER BY score
        LIMIT ?
        """,
        params,
    ).fetchall()
    return [_row_to_hit(r, float(r["score"])) for r in rows]


def _vec_search(
    conn: sqlite3.Connection,
    task: str,
    *,
    category: str | None,
    limit: int,
) -> list[SkillHit]:
    _ensure_vec_table(conn)
    count = conn.execute("SELECT count(*) AS c FROM skills_vec").fetchone()["c"]
    if count == 0:
        return []
    qvec = serialize_float32(embed_query(task))
    # Oversample then filter category in Python (vec0 + JOIN filters are finicky)
    k = limit * 4 if category else limit
    k = max(1, min(int(k), 80))
    rows = conn.execute(
        """
        SELECT s.name, s.path, s.primary_cat, s.description, s.skill_md,
               v.distance AS score
        FROM skills_vec v
        JOIN skills s ON s.rowid = v.rowid
        WHERE v.embedding MATCH ?
          AND k = ?
        ORDER BY v.distance
        """,
        [qvec, k],
    ).fetchall()
    hits = [_row_to_hit(r, float(r["score"])) for r in rows]
    if category:
        hits = [h for h in hits if h.primary == category]
    return hits[:limit]


def _rrf_fuse(lists: list[list[SkillHit]]) -> list[SkillHit]:
    scores: dict[str, float] = {}
    by_name: dict[str, SkillHit] = {}
    for hits in lists:
        for rank, h in enumerate(hits, start=1):
            scores[h.name] = scores.get(h.name, 0.0) + 1.0 / (RRF_K + rank)
            by_name[h.name] = h
    ordered = sorted(scores.keys(), key=lambda n: -scores[n])
    out: list[SkillHit] = []
    for name in ordered:
        h = by_name[name]
        # Negate so lower-is-better matches bm25 sort convention
        out.append(
            SkillHit(h.name, h.path, h.primary, h.description, -scores[name], h.skill_md)
        )
    return out


# Task cues → preferred primary categories (soft boost)
_CUE_CATS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(reticle|playwright|e2e|tdd|verify ui|flaky|xctest)\b", re.I), "testing"),
    (re.compile(r"\b(react|next\.?js|vue|svelte|tailwind|frontend)\b", re.I), "frontend"),
    (re.compile(r"\b(figma|impeccable|design taste|visual design|ui/?ux|landing.*(look|design))\b", re.I), "ui-ux"),
    (re.compile(r"\b(api|postgres|graphql|backend|fastapi|django|apache.?age|ag_catalog)\b", re.I), "backend"),
    (re.compile(r"\b(graphrag|rag.?pipeline|vector.?search|sqlite.?vec|hybrid.?retriev|embedding.?model)\b", re.I), "agents-ai"),
    (re.compile(r"\b(seo|serp|backlink|schema\.org)\b", re.I), "seo"),
    (re.compile(r"\b(ios|swift|android|flutter|react.?native|speech.?analyzer|foundation.?models)\b", re.I), "mobile"),
]


def _boost_by_task_cues(task: str, hits: list[SkillHit]) -> list[SkillHit]:
    preferred = [cat for rx, cat in _CUE_CATS if rx.search(task)]
    if not preferred or not hits:
        return hits
    boosted: list[SkillHit] = []
    for h in hits:
        score = h.score
        if h.primary in preferred:
            score -= 0.05 * (1 + preferred.index(h.primary) * 0.15)
        if preferred and any(p in {"frontend", "ui-ux", "testing"} for p in preferred):
            if re.search(r"desktop|electron|tauri", f"{h.name} {h.description}", re.I):
                score += 0.08
        if re.search(r"\b(react|next\.?js)\b", task, re.I):
            if re.search(r"webflow|wordpress|shopify", f"{h.name} {h.description}", re.I):
                score += 0.06
            if re.search(r"^(senior-frontend|react-|frontend-developer|typescript-pro)", h.name, re.I):
                score -= 0.04
        boosted.append(SkillHit(h.name, h.path, h.primary, h.description, score, h.skill_md))
    boosted.sort(key=lambda x: x.score)
    return boosted


def search(
    task: str,
    *,
    category: str | None = None,
    top_k: int = 5,
    db_path: Path = DEFAULT_DB,
    mode: str = "hybrid",
    rerank: bool = True,
) -> list[SkillHit]:
    """Search skills. mode: hybrid | fts | vec. Intent rewrite + optional cross-encoder."""
    intent = parse_intent(task)
    return search_with_intent(
        intent,
        category=category,
        top_k=top_k,
        db_path=db_path,
        mode=mode,
        rerank=rerank,
    )[1]


def search_with_intent(
    intent: Intent,
    *,
    category: str | None = None,
    top_k: int = 5,
    db_path: Path = DEFAULT_DB,
    mode: str = "hybrid",
    rerank: bool = True,
) -> tuple[Intent, list[SkillHit]]:
    """Full pipeline: intent → FTS/vec on rewrite → RRF → cue boost → cross-encoder."""
    ensure_index(db_path=db_path)
    # Soft category: user override wins; else leave open (multi-domain tasks)
    cat = category
    query = intent.rewrite or intent.original
    original = intent.original or query
    conn = _connect(db_path)
    fetch_k = max(top_k * 5, 25) if not cat else max(top_k * 3, 15)
    fetch_k = max(1, min(int(fetch_k), 80))
    try:
        fts_hits: list[SkillHit] = []
        vec_hits: list[SkillHit] = []
        if mode in {"hybrid", "fts"}:
            # Elastic: original must + enrich should → FTS OR of both token sets
            fts_hits = _fts_search(conn, query, category=cat, limit=fetch_k)
        if mode in {"hybrid", "vec"}:
            try:
                if has_vectors(conn):
                    vec_hits = _vec_search(conn, query, category=cat, limit=fetch_k)
                else:
                    vec_hits = []
            except Exception:
                vec_hits = []
        if mode == "fts" or (mode == "hybrid" and not vec_hits):
            hits = fts_hits
        elif mode == "vec" or (mode == "hybrid" and not fts_hits):
            hits = vec_hits
        else:
            hits = _rrf_fuse([fts_hits, vec_hits])
    finally:
        conn.close()

    # Prefer intent categories in cue boost (merge with regex cues on original)
    hits = _boost_by_task_cues(original, hits)
    if intent.categories:
        hits = _boost_intent_categories(intent.categories, hits)

    limit = max(1, min(int(top_k), 25))
    if rerank and mode == "hybrid":
        hits = cross_encoder_rerank(original, hits, top_k=limit)
    else:
        hits = hits[:limit]
    return intent, hits


def _boost_intent_categories(categories: list[str], hits: list[SkillHit]) -> list[SkillHit]:
    if not categories or not hits:
        return hits
    boosted: list[SkillHit] = []
    for h in hits:
        score = h.score
        if h.primary in categories:
            score -= 0.06 * (1 + categories.index(h.primary) * 0.12)
        boosted.append(SkillHit(h.name, h.path, h.primary, h.description, score, h.skill_md))
    boosted.sort(key=lambda x: x.score)
    return boosted


def get_skill(name: str, db_path: Path = DEFAULT_DB) -> SkillHit | None:
    ensure_index(db_path=db_path)
    conn = _connect(db_path)
    r = conn.execute(
        "SELECT name, path, primary_cat, description, skill_md FROM skills WHERE name = ?",
        (name,),
    ).fetchone()
    if not r:
        r = conn.execute(
            "SELECT name, path, primary_cat, description, skill_md FROM skills WHERE lower(name) = lower(?)",
            (name,),
        ).fetchone()
    conn.close()
    if not r:
        return None
    return _row_to_hit(r, 0.0)


def list_skills(
    *,
    category: str | None = None,
    limit: int = 50,
    db_path: Path = DEFAULT_DB,
) -> list[dict]:
    ensure_index(db_path=db_path)
    conn = _connect(db_path)
    limit = max(1, min(int(limit), 200))
    if category:
        rows = conn.execute(
            """
            SELECT name, path, primary_cat, description
            FROM skills WHERE primary_cat = ?
            ORDER BY name LIMIT ?
            """,
            (category, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT name, path, primary_cat, description
            FROM skills ORDER BY name LIMIT ?
            """,
            (limit,),
        ).fetchall()
    conn.close()
    return [
        {
            "name": r["name"],
            "path": host_skill_path(r["path"]),
            "skill_md_path": f"{host_skill_path(r['path'])}/SKILL.md",
            "primary": r["primary_cat"],
            "description": r["description"][:280],
        }
        for r in rows
    ]


def compose(
    task: str,
    *,
    top_k: int = 5,
    db_path: Path = DEFAULT_DB,
) -> dict:
    """Primary + supporting skills across distinct categories (intent-aware)."""
    intent, hits = search_with_intent(
        parse_intent(task),
        top_k=max(top_k * 6, 24),
        db_path=db_path,
        mode="hybrid",
        rerank=True,
    )
    if not hits:
        return {"primary": None, "supporting": [], "task": task, "intent": intent.as_dict()}
    preferred = intent.categories or [cat for rx, cat in _CUE_CATS if rx.search(task)]
    preferred_set = set(preferred)
    # Global rerank order, first hit whose category is an intent domain
    primary = next((h for h in hits if h.primary in preferred_set), hits[0]) if preferred_set else hits[0]

    supporting: list[SkillHit] = []
    seen_cats = {primary.primary}
    seen_names = {primary.name}

    for cat in preferred:
        if cat in seen_cats:
            continue
        cand = next((h for h in hits if h.primary == cat and h.name not in seen_names), None)
        if cand:
            supporting.append(cand)
            seen_cats.add(cat)
            seen_names.add(cand.name)
        if len(supporting) >= max(0, top_k - 1):
            break

    for h in hits:
        if len(supporting) >= max(0, top_k - 1):
            break
        if h.name in seen_names:
            continue
        if h.primary in seen_cats:
            continue
        supporting.append(h)
        seen_cats.add(h.primary)
        seen_names.add(h.name)

    if len(supporting) < max(0, top_k - 1):
        for h in hits:
            if h.name in seen_names:
                continue
            supporting.append(h)
            seen_names.add(h.name)
            if len(supporting) >= max(0, top_k - 1):
                break

    def pack(h: SkillHit) -> dict:
        return {
            "name": h.name,
            "path": host_skill_path(h.path),
            "primary": h.primary,
            "score": round(h.score, 4),
            "description": h.description[:320],
            "skill_md_path": f"{host_skill_path(h.path)}/SKILL.md",
        }

    return {
        "task": task,
        "intent": intent.as_dict(),
        "primary": pack(primary),
        "supporting": [pack(s) for s in supporting],
        "mode": "intent+fts5+sqlite-vec+rrf+rerank",
        "instruction": (
            "Read primary SKILL.md first and follow it. "
            "Load supporting skills only if the primary skill requires them or the task clearly spans those domains."
        ),
    }
