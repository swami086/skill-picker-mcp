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


def _env_path(key: str, default: Path) -> Path:
    raw = os.environ.get(key, "").strip()
    return Path(raw).expanduser() if raw else default


DEFAULT_SKILLS_ROOT = _env_path(
    "SKILL_PICKER_SKILLS_ROOT",
    Path.home() / ".cursor" / "skills",
)
DEFAULT_DB = _env_path(
    "SKILL_PICKER_DB",
    Path.home() / ".cursor" / "skill-picker-mcp" / "data" / "skills.db",
)


def resolve_catalog_path(*, skills_root: Path | None = None) -> Path:
    """Prefer explicit env, then writable /data catalog, then skills-root catalog."""
    explicit = os.environ.get("SKILL_PICKER_CATALOG", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    for cand in (
        Path("/data/skills-catalog.json"),
        Path.home() / ".cursor" / "skill-picker-mcp" / "data" / "skills-catalog.json",
    ):
        if cand.is_file():
            return cand
    root = skills_root or DEFAULT_SKILLS_ROOT
    return root / "skills-catalog.json"


DEFAULT_CATALOG = resolve_catalog_path()
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
    pack: str = ""


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
          body TEXT NOT NULL,
          pack TEXT NOT NULL DEFAULT ''
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS skills_fts USING fts5(
          name,
          primary_cat,
          description,
          tags,
          body,
          pack,
          content='skills',
          content_rowid='rowid'
        );
        """
    )
    cols = {r[1] for r in conn.execute("PRAGMA table_info(skills)").fetchall()}
    if "pack" not in cols:
        conn.execute("ALTER TABLE skills ADD COLUMN pack TEXT NOT NULL DEFAULT ''")
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


def _embed_doc(name: str, primary: str, desc: str, body: str, pack: str = "") -> str:
    pack_bit = f" pack:{pack}" if pack else ""
    return f"{name}\n[{primary}{pack_bit}]\n{desc}\n{body[:EMBED_CHARS]}"


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
    """Rescan skills packages → rewrite skills-catalog.json (+ openmemory jsonl).

    Writes to skills_root when writable; otherwise SKILL_PICKER_CATALOG_OUT or /data
    (Docker RO /skills mount).

    If the in-container scan finds far fewer skills than an existing host catalog
    (common when many packages are symlinks outside the bind mount), prefer the
    host catalog so indexing stays complete.
    """
    import shutil

    script = skills_root / "scripts" / "build-skills-catalog.py"
    if not script.is_file():
        return {"ok": False, "error": f"missing catalog builder: {script}"}

    catalog_out = os.environ.get("SKILL_PICKER_CATALOG_OUT", "").strip()
    if not catalog_out:
        if Path("/data").is_dir() and os.access("/data", os.W_OK):
            catalog_out = "/data"
        else:
            catalog_out = str(skills_root)

    env = os.environ.copy()
    env["SKILL_PICKER_CATALOG_OUT"] = catalog_out
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(skills_root),
            "--catalog-out",
            catalog_out,
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env=env,
    )
    catalog_path = Path(catalog_out) / "skills-catalog.json"
    host_catalog = skills_root / "skills-catalog.json"
    preferred = catalog_path
    note = None
    if proc.returncode == 0 and catalog_path.is_file() and host_catalog.is_file():
        n_new = len(catalog_skill_names(catalog_path))
        n_host = len(catalog_skill_names(host_catalog))
        # Symlink-heavy host trees look tiny inside Docker bind mounts.
        if n_host > max(n_new + 50, int(n_new * 1.25)):
            try:
                shutil.copy2(host_catalog, catalog_path)
                preferred = catalog_path
                note = (
                    f"preferred host catalog ({n_host} skills) over in-container "
                    f"scan ({n_new}); copied to {catalog_path}"
                )
            except OSError as exc:
                preferred = host_catalog
                note = f"host catalog preferred ({n_host} vs {n_new}); using {host_catalog} ({exc})"
    elif proc.returncode != 0 and host_catalog.is_file():
        preferred = host_catalog
        note = "catalog build failed; falling back to existing host catalog"

    out = {
        "ok": preferred.is_file(),
        "returncode": proc.returncode,
        "script": str(script),
        "catalog_out": catalog_out,
        "catalog_path": str(preferred),
        "stdout_tail": (proc.stdout or "")[-800:],
    }
    if note:
        out["note"] = note
    if proc.returncode != 0:
        out["stderr_tail"] = (proc.stderr or "")[-800:]
    if preferred.is_file():
        out["total"] = len(catalog_skill_names(preferred))
        out["openmemory_jsonl"] = str(Path(catalog_out) / "skills-catalog.openmemory.jsonl")
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
        catalog = Path(catalog_info["catalog_path"])
        # Keep env in sync for subsequent resolve_catalog_path calls
        os.environ["SKILL_PICKER_CATALOG"] = str(catalog)
    else:
        catalog = resolve_catalog_path(skills_root=skills_root)

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
          name, primary_cat, description, tags, body, pack,
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
        pack = str(s.get("pack") or "")
        desc = str(s.get("description") or "")
        tags = ",".join(s.get("tags") or [])
        md = _read_skill_md(skills_root, rel)
        body = _body_from_md(md) if md else desc
        rows_out.append((name, rel, primary, desc, tags, md, body, pack))
        embed_inputs.append(_embed_doc(name, primary, desc, body, pack=pack))

    for row in rows_out:
        conn.execute(
            """
            INSERT INTO skills
              (name, path, primary_cat, description, tags, skill_md, body, pack)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            row,
        )
    conn.execute(
        """
        INSERT INTO skills_fts(rowid, name, primary_cat, description, tags, body, pack)
        SELECT rowid, name, primary_cat, description, tags, body, pack FROM skills
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
    pack = ""
    try:
        pack = r["pack"] or ""
    except (KeyError, IndexError):
        pack = ""
    return SkillHit(
        name=r["name"],
        path=r["path"],
        primary=r["primary_cat"],
        description=r["description"],
        score=score,
        skill_md=r["skill_md"] or "",
        pack=pack,
    )


def _clone_hit(h: SkillHit, score: float) -> SkillHit:
    return SkillHit(h.name, h.path, h.primary, h.description, score, h.skill_md, h.pack)


def _fts_search(
    conn: sqlite3.Connection,
    task: str,
    *,
    category: str | None,
    pack: str | None,
    limit: int,
) -> list[SkillHit]:
    q = _fts_query(task)
    params: list = [q]
    where = "skills_fts MATCH ?"
    if category:
        where += " AND s.primary_cat = ?"
        params.append(category)
    if pack:
        where += " AND s.pack = ?"
        params.append(pack)
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT s.name, s.path, s.primary_cat, s.description, s.skill_md, s.pack,
               bm25(skills_fts, 12.0, 6.0, 8.0, 3.0, 1.0, 4.0) AS score
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
    pack: str | None,
    limit: int,
) -> list[SkillHit]:
    _ensure_vec_table(conn)
    count = conn.execute("SELECT count(*) AS c FROM skills_vec").fetchone()["c"]
    if count == 0:
        return []
    qvec = serialize_float32(embed_query(task))
    # Oversample then filter in Python (vec0 + JOIN filters are finicky)
    mult = 1
    if category:
        mult *= 4
    if pack:
        mult *= 4
    k = limit * mult if mult > 1 else limit
    k = max(1, min(int(k), 120))
    rows = conn.execute(
        """
        SELECT s.name, s.path, s.primary_cat, s.description, s.skill_md, s.pack,
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
    if pack:
        hits = [h for h in hits if h.pack == pack]
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
        out.append(_clone_hit(h, -scores[name]))
    return out


# Task cues → preferred primary categories (soft boost)
_CUE_CATS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(reticle|playwright|e2e|tdd|verify ui|flaky|xctest)\b", re.I), "testing"),
    (re.compile(r"\b(react|next\.?js|vue|svelte|tailwind|frontend)\b", re.I), "frontend"),
    (re.compile(r"\b(figma|impeccable|design taste|visual design|ui/?ux|landing.*(look|design))\b", re.I), "ui-ux"),
    # security before backend — "OWASP … API" must not drown in backend
    (
        re.compile(
            r"\b(owasp|threat.?model|pen.?test|secops|ciso|security.?review|vuln|"
            r"appsec|infosec|zero.?trust)\b",
            re.I,
        ),
        "security",
    ),
    (re.compile(r"\b(api|postgres|graphql|backend|fastapi|django|apache.?age|ag_catalog)\b", re.I), "backend"),
    (re.compile(r"\b(graphrag|rag.?pipeline|vector.?search|sqlite.?vec|hybrid.?retriev|embedding.?model)\b", re.I), "agents-ai"),
    (re.compile(r"\b(seo|serp|backlink|schema\.org|aeo|answer.?engine)\b", re.I), "seo"),
    (re.compile(r"\b(ios|swift|android|flutter|react.?native|speech.?analyzer|foundation.?models)\b", re.I), "mobile"),
    (
        re.compile(
            r"\b(cfo|runway|unit.?economics|dilution|fundraising|board.?deck|boardroom)\b",
            re.I,
        ),
        "business",
    ),
]


def _boost_by_task_cues(task: str, hits: list[SkillHit]) -> list[SkillHit]:
    preferred = [cat for rx, cat in _CUE_CATS if rx.search(task)]
    # Security intent beats incidental "api"/"backend" cues
    if "security" in preferred:
        preferred = [c for c in preferred if c != "backend"]
    if not preferred or not hits:
        return hits
    boosted: list[SkillHit] = []
    for h in hits:
        score = h.score
        if h.primary in preferred:
            score -= 0.05 * (1 + preferred.index(h.primary) * 0.15)
        # Name/description keyword boost for security / finance when cued
        blob = f"{h.name} {h.description}"
        if "security" in preferred and re.search(
            r"security|ciso|owasp|threat|secops|appsec", blob, re.I
        ):
            score -= 0.04
        if "business" in preferred and re.search(
            r"\b(cfo|runway|board|fundraising|dilution|unit.?economics)\b", blob, re.I
        ):
            score -= 0.03
        if preferred and any(p in {"frontend", "ui-ux", "testing"} for p in preferred):
            if re.search(r"desktop|electron|tauri", blob, re.I):
                score += 0.08
        if re.search(r"\b(react|next\.?js)\b", task, re.I):
            if re.search(r"webflow|wordpress|shopify", blob, re.I):
                score += 0.06
            if re.search(r"^(senior-frontend|react-|frontend-developer|typescript-pro)", h.name, re.I):
                score -= 0.04
        boosted.append(_clone_hit(h, score))
    boosted.sort(key=lambda x: x.score)
    return boosted


def search(
    task: str,
    *,
    category: str | None = None,
    pack: str | None = None,
    top_k: int = 5,
    db_path: Path = DEFAULT_DB,
    mode: str = "hybrid",
) -> list[SkillHit]:
    """Search skills. mode: hybrid | fts | vec."""
    ensure_index(db_path=db_path)
    conn = _connect(db_path)
    fetch_k = max(top_k * 5, 25) if not (category or pack) else max(top_k * 3, 15)
    fetch_k = max(1, min(int(fetch_k), 80))
    try:
        fts_hits: list[SkillHit] = []
        vec_hits: list[SkillHit] = []
        if mode in {"hybrid", "fts"}:
            fts_hits = _fts_search(
                conn, task, category=category, pack=pack, limit=fetch_k
            )
        if mode in {"hybrid", "vec"}:
            try:
                if has_vectors(conn):
                    vec_hits = _vec_search(
                        conn, task, category=category, pack=pack, limit=fetch_k
                    )
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
    return _boost_by_task_cues(task, hits)[: max(1, min(int(top_k), 25))]


def get_skill(name: str, db_path: Path = DEFAULT_DB) -> SkillHit | None:
    ensure_index(db_path=db_path)
    conn = _connect(db_path)
    r = conn.execute(
        "SELECT name, path, primary_cat, description, skill_md, pack FROM skills WHERE name = ?",
        (name,),
    ).fetchone()
    if not r:
        r = conn.execute(
            "SELECT name, path, primary_cat, description, skill_md, pack FROM skills WHERE lower(name) = lower(?)",
            (name,),
        ).fetchone()
    conn.close()
    if not r:
        return None
    return _row_to_hit(r, 0.0)


def list_skills(
    *,
    category: str | None = None,
    pack: str | None = None,
    limit: int = 50,
    db_path: Path = DEFAULT_DB,
) -> list[dict]:
    ensure_index(db_path=db_path)
    conn = _connect(db_path)
    limit = max(1, min(int(limit), 200))
    clauses: list[str] = []
    params: list = []
    if category:
        clauses.append("primary_cat = ?")
        params.append(category)
    if pack:
        clauses.append("pack = ?")
        params.append(pack)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT name, path, primary_cat, description, pack
        FROM skills {where}
        ORDER BY name LIMIT ?
        """,
        params,
    ).fetchall()
    conn.close()
    return [
        {
            "name": r["name"],
            "path": host_skill_path(r["path"]),
            "skill_md_path": f"{host_skill_path(r['path'])}/SKILL.md",
            "primary": r["primary_cat"],
            "pack": r["pack"] or "",
            "description": r["description"][:280],
        }
        for r in rows
    ]


def rank_authors(hits: list[SkillHit]) -> list[dict]:
    """Rank author packs by best (lowest) member score. Unpacked skills omitted."""
    best: dict[str, SkillHit] = {}
    for h in hits:
        if not h.pack:
            continue
        cur = best.get(h.pack)
        if cur is None or h.score < cur.score:
            best[h.pack] = h
    ordered = sorted(best.items(), key=lambda kv: kv[1].score)
    out: list[dict] = []
    for pack, h in ordered:
        out.append(
            {
                "pack": pack,
                "router": pack,  # convention: router skill name == pack id
                "best_skill": h.name if h.name != pack else None,
                "best_skill_or_router": h.name,
                "category": h.primary,
                "score": round(h.score, 4),
            }
        )
    # Prefer best_skill to be a member when router also present
    for a in out:
        pack = a["pack"]
        members = [h for h in hits if h.pack == pack and h.name != pack]
        if members:
            top = min(members, key=lambda x: x.score)
            a["best_skill"] = top.name
            a["score"] = round(top.score, 4)
            a["category"] = top.primary
        elif a["best_skill"] == pack:
            a["best_skill"] = None
    return out


def order_hits_router_first(hits: list[SkillHit]) -> list[SkillHit]:
    """Reorder search hits so pack routers lead (find_helpful_skills entrypoints).

    When authors are present, inject/synthesize each pack router ahead of members.
    Unpacked skills keep relative order after routers.
    """
    if not hits:
        return []
    authors = rank_authors(hits)
    if not authors:
        return list(hits)
    out: list[SkillHit] = []
    seen: set[str] = set()
    for a in authors:
        router = _router_hit_for_pack(
            a["pack"],
            hits,
            fallback_primary=a["category"],
            fallback_score=a["score"],
        )
        if router.name not in seen:
            out.append(router)
            seen.add(router.name)
    for h in hits:
        if h.name not in seen:
            out.append(h)
            seen.add(h.name)
    return out


def _router_hit_for_pack(
    pack: str,
    hits: list[SkillHit],
    *,
    fallback_primary: str,
    fallback_score: float,
) -> SkillHit:
    """Return the pack router hit from results, or synthesize a router entrypoint."""
    for h in hits:
        if h.pack == pack and h.name == pack:
            return h
    # Synthesize — agent loads ~/.cursor/skills/<pack>/SKILL.md
    return SkillHit(
        name=pack,
        path=pack,
        primary=fallback_primary,
        description=f"Router for pack {pack}. Load this first, then one member skill.",
        score=fallback_score,
        skill_md="",
        pack=pack,
    )


def _best_member(pack: str, hits: list[SkillHit]) -> SkillHit | None:
    members = [h for h in hits if h.pack == pack and h.name != pack]
    if not members:
        # If only router hit, no member
        return None
    return min(members, key=lambda h: h.score)


def compose_from_hits(
    task: str,
    hits: list[SkillHit],
    *,
    top_k: int = 5,
) -> dict:
    """Compose primary + supporting from ranked hits — router-first, multi-pack.

    Pure function for TDD; `compose()` delegates here after search.
    """
    if not hits:
        return {
            "task": task,
            "primary": None,
            "supporting": [],
            "packs_used": [],
            "authors": [],
            "route": {
                "step1": "Load router SKILL.md for the chosen author pack",
                "step2": "Let the router pick one member skill",
                "step3": "Repeat for additional packs in authors[]",
            },
            "mode": "fts5+sqlite-vec+rrf",
            "instruction": (
                "Always enter via the pack router. Never start from a random pack member. "
                "Load supporting pack routers when the task spans authors."
            ),
        }

    preferred = [cat for rx, cat in _CUE_CATS if rx.search(task)]
    seed = hits[0]
    if preferred and seed.primary not in preferred:
        for cat in preferred:
            cand = next((h for h in hits if h.primary == cat), None)
            if cand:
                seed = cand
                break

    authors = rank_authors(hits)
    # Prefer a ranked author pack over an unpacked seed (router-first policy)
    if not seed.pack and authors:
        top = authors[0]
        seed = next(
            (h for h in hits if h.pack == top["pack"]),
            SkillHit(
                name=top["pack"],
                path=top["pack"],
                primary=top["category"],
                description=f"Router for {top['pack']}",
                score=top["score"],
                skill_md="",
                pack=top["pack"],
            ),
        )

    # If seed has a pack, primary entrypoint is that pack's router
    if seed.pack:
        primary_hit = _router_hit_for_pack(
            seed.pack,
            hits,
            fallback_primary=seed.primary,
            fallback_score=seed.score,
        )
        recommended = _best_member(seed.pack, hits) or (
            seed if seed.name != seed.pack else None
        )
    else:
        primary_hit = seed
        recommended = None

    supporting: list[tuple[SkillHit, SkillHit | None]] = []  # (router_or_hit, recommended)
    seen_packs = {primary_hit.pack} if primary_hit.pack else set()
    seen_names = {primary_hit.name}
    if recommended:
        seen_names.add(recommended.name)

    # Prefer other author packs from rank_authors
    for author in authors:
        if len(supporting) >= max(0, top_k - 1):
            break
        pack = author["pack"]
        if pack in seen_packs:
            continue
        router = _router_hit_for_pack(
            pack,
            hits,
            fallback_primary=author["category"],
            fallback_score=author["score"],
        )
        member = _best_member(pack, hits)
        supporting.append((router, member))
        seen_packs.add(pack)
        seen_names.add(router.name)
        if member:
            seen_names.add(member.name)

    # Fill with category-diverse unpacked / remaining hits if needed
    for h in hits:
        if len(supporting) >= max(0, top_k - 1):
            break
        if h.name in seen_names:
            continue
        if h.pack and h.pack in seen_packs:
            continue
        if h.pack:
            router = _router_hit_for_pack(
                h.pack, hits, fallback_primary=h.primary, fallback_score=h.score
            )
            member = _best_member(h.pack, hits) or h
            supporting.append((router, member if member.name != router.name else None))
            seen_packs.add(h.pack)
            seen_names.add(router.name)
        else:
            supporting.append((h, None))
            seen_names.add(h.name)

    def hit_payload(
        h: SkillHit,
        *,
        role: str,
        recommended_skill: str | None = None,
    ) -> dict:
        out = {
            "name": h.name,
            "path": host_skill_path(h.path),
            "primary": h.primary,
            "pack": h.pack or "",
            "role": role,
            "score": round(h.score, 4),
            "description": h.description[:320],
            "skill_md_path": f"{host_skill_path(h.path)}/SKILL.md",
        }
        if recommended_skill:
            out["recommended_skill"] = recommended_skill
            out["recommended_skill_md_path"] = (
                f"{host_skill_path(recommended_skill)}/SKILL.md"
            )
        return out

    primary_payload = hit_payload(
        primary_hit,
        role="router" if primary_hit.pack and primary_hit.name == primary_hit.pack else "skill",
        recommended_skill=recommended.name if recommended else None,
    )

    support_payloads = []
    for router_or_hit, member in supporting:
        is_router = bool(router_or_hit.pack and router_or_hit.name == router_or_hit.pack)
        support_payloads.append(
            hit_payload(
                router_or_hit,
                role="router" if is_router else "skill",
                recommended_skill=member.name if member else None,
            )
        )

    packs_used = sorted(
        ({primary_hit.pack} | {s.pack for s, _ in supporting if s.pack}) - {""}
    )

    return {
        "task": task,
        "primary": primary_payload,
        "supporting": support_payloads,
        "packs_used": packs_used,
        "authors": authors,
        "route": {
            "step1": "Load router SKILL.md for primary.pack (or primary.name if role=router)",
            "step2": "Follow the router to primary.recommended_skill (one member only)",
            "step3": "For each supporting author: load that router, then its recommended_skill",
        },
        "mode": "fts5+sqlite-vec+rrf",
        "instruction": (
            "Always enter via the pack router (role=router). "
            "Then load recommended_skill from that pack — never bulk-load a pack. "
            "When packs_used has multiple authors, load each supporting router the same way."
        ),
    }


def compose(
    task: str,
    *,
    top_k: int = 5,
    db_path: Path = DEFAULT_DB,
) -> dict:
    """Primary + supporting skills across distinct categories AND packs (router-first)."""
    hits = search(task, top_k=max(top_k * 6, 24), db_path=db_path)
    return compose_from_hits(task, hits, top_k=top_k)
