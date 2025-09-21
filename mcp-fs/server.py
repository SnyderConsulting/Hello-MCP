#!/usr/bin/env python3
"""Read/write filesystem MCP server with job controls."""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import logging
import mimetypes
import os
import re
import shlex
import shutil
import signal
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence

import hashlib

from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from starlette.routing import Mount, Route
import uvicorn

ENABLE_WRITE = os.environ.get("MCP_ENABLE_WRITE", "0") == "1"
ENABLE_EXEC = os.environ.get("MCP_ENABLE_EXEC", "0") == "1"


def _configure_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    level_name = os.environ.get("MCP_FS_LOG_LEVEL")
    if level_name:
        level = getattr(logging, level_name.upper(), None)
        if isinstance(level, int):
            logger.setLevel(level)
            if not logging.getLogger().handlers:
                logging.basicConfig(level=level)
        else:
            logger.warning(
                "Unknown MCP_FS_LOG_LEVEL %r. Falling back to default levels.",
                level_name,
            )
    return logger


logger = _configure_logger()

def _int_from_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "Invalid integer for %s: %r. Falling back to %d.",
            name,
            raw,
            default,
        )
        return default


SEARCH_MAX_FILE_BYTES = _int_from_env("MCP_FS_SEARCH_MAX_FILE_BYTES", 0)


def _env_path(key: str, default: str) -> str:
    raw = os.environ.get(key, default).strip()
    rel = Path(raw)
    if str(rel) in {"", "."}:
        return "."
    return str(rel)


LOG_SINK_REL = _env_path("MCP_LOG_SINK", "hello-mcp/logs")
NOTES_ROOT_REL = _env_path("MCP_NOTES_ROOT", "hello-mcp/notes")
TRASH_ROOT_REL = _env_path("MCP_TRASH_ROOT", ".mcp_trash")


# ------------------------ FS helpers (read/write jail) ------------------------


def resolve_base(base: str) -> Path:
    p = Path(base).expanduser().resolve()
    if not p.exists() or not p.is_dir():
        raise ValueError(f"FS root does not exist or is not a directory: {p}")
    return p


def safe_join(base: Path, user_path: str) -> Path:
    candidate = (base / user_path).resolve()
    if os.path.commonpath([str(candidate), str(base)]) != str(base):
        raise ValueError("Path escapes FS root")
    return candidate


def _resolve_path(path: str) -> Path:
    return safe_join(FS_ROOT, str(Path(path)))


def is_probably_text(sample: bytes) -> bool:
    if b"\x00" in sample:
        return False
    try:
        sample.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def read_text_safely(p: Path, max_bytes: int) -> Dict[str, Any]:
    with p.open("rb") as f:
        data = f.read(max_bytes + 1)
    truncated = len(data) > max_bytes
    sample = data[:4096]
    if is_probably_text(sample):
        try:
            text = data[:max_bytes].decode("utf-8", errors="replace")
        except Exception:
            text = data[:max_bytes].decode("latin-1", errors="replace")
        return {"kind": "text", "text": text, "truncated": truncated}
    return {
        "kind": "bytes",
        "base64": base64.b64encode(data[:max_bytes]).decode("ascii"),
        "truncated": truncated,
    }


class TextReadResult(NamedTuple):
    text: Optional[str]
    error: Optional[BaseException]
    is_binary: bool


def _read_text_limited(path: Path, max_bytes: Optional[int]) -> TextReadResult:
    """Read up to ``max_bytes`` from ``path`` as text.

    Returns ``TextReadResult`` with the decoded text when successful. When the
    file cannot be read or appears binary, the result contains ``None`` for the
    text and flags whether it was because of a binary detection or an error.
    A ``max_bytes`` value of ``None`` or ``<= 0`` disables the limit."""

    limit = max_bytes if max_bytes and max_bytes > 0 else None
    try:
        with path.open("rb") as fh:
            data = fh.read() if limit is None else fh.read(limit)
    except OSError as exc:
        return TextReadResult(None, exc, False)

    sample = data[:4096]
    if not is_probably_text(sample):
        return TextReadResult(None, None, True)

    try:
        text = data.decode("utf-8", errors="ignore")
    except Exception:
        text = data.decode("latin-1", errors="ignore")
    return TextReadResult(text, None, False)


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    if isinstance(value, (int, float)):
        return value != 0
    return bool(value)


def _error(code: str, message: str, **extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"error": {"code": code, "message": message}}
    if extra:
        payload["error"].update(extra)
    return payload


def _file_metadata(p: Path) -> Dict[str, Any]:
    st = p.stat()
    mime, _ = mimetypes.guess_type(str(p))
    return {
        "path": str(p.relative_to(FS_ROOT)),
        "name": p.name,
        "is_dir": p.is_dir(),
        "size": st.st_size,
        "mtime": st.st_mtime,
        "mime": mime or ("inode/directory" if p.is_dir() else "application/octet-stream"),
    }


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _append_jsonl_atomic(rel_path: str, record: Dict[str, Any], create_parents: bool = True) -> int:
    target = _resolve_path(rel_path)
    if create_parents:
        _ensure_dir(target.parent)
    tmp = target.with_name(f"{target.name}.{uuid.uuid4().hex}.tmp")
    line = json.dumps(record, ensure_ascii=False)
    data = (line + "\n").encode("utf-8")
    try:
        with tmp.open("wb") as f:
            if target.exists():
                with target.open("rb") as src:
                    shutil.copyfileobj(src, f)
            f.write(data)
        tmp.replace(target)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return len(data)


# ----------------------------- MCP server setup -----------------------------

mcp = FastMCP(
    name="Hello MCP FS",
    instructions=(
        "Browse, edit, and manage files inside a jailed workspace. "
        "Run `init()` first to discover the log sink, notes root, and starter commands."
    ),
)

FS_ROOT: Path = Path(".")

# ------------------------------- init tool ----------------------------------

DEFAULT_JOB_TEMPLATES = [
    {
        "name": "tee_to_log_sink",
        "description": "Run any script and tee stdout/stderr into the shared log sink.",
        "command": 'bash -lc "python your_script.py 2>&1 | tee \\"{logs}/run.log\\""',
    }
]


def _log_sink_path() -> Path:
    return _resolve_path(LOG_SINK_REL)


def _notes_root_path() -> Path:
    return _resolve_path(NOTES_ROOT_REL)


def _trash_root_path() -> Path:
    return _resolve_path(TRASH_ROOT_REL)


def _gather_agents_docs() -> List[Dict[str, str]]:
    """Collect AGENTS.md files under FS_ROOT and return their contents."""

    try:
        candidates = sorted(FS_ROOT.rglob("AGENTS.md"))
    except Exception:
        return []

    docs: List[Dict[str, str]] = []
    for path in candidates:
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        try:
            rel = str(path.relative_to(FS_ROOT))
        except ValueError:
            rel = str(path)
        docs.append({"path": rel, "content": content})
    return docs


@mcp.tool()
def init(mode: str = "quick", include: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Initialize the session and surface orientation info.
    Run this first to confirm env + log sink; tip: share paths.logs with teammates.
    Usage: init(mode="quick", include=["quick_commands","paths","docs"])
    Params: mode:"quick"|"full", include:list[str]
    Returns: {markdown, paths:{logs,notes,jobs}, env:{write_enabled,exec_enabled}}
    Gotchas: When writes are disabled, folder creation is skipped."""

    normalized_mode = (mode or "quick").lower()
    if normalized_mode not in {"quick", "full"}:
        normalized_mode = "quick"

    requested_sections = list(include) if include else ["quick_commands", "paths", "docs"]

    log_sink_rel = str(Path(LOG_SINK_REL))
    notes_rel = str(Path(NOTES_ROOT_REL))
    jobs_rel = str((_jobs_root()).relative_to(FS_ROOT))

    if ENABLE_WRITE:
        for ensure_path in (_log_sink_path(), _notes_root_path(), _trash_root_path()):
            try:
                _ensure_dir(ensure_path)
            except Exception:
                pass

    sections: Dict[str, str] = {}

    quick_lines = [
        "## Quick commands",
        "- `gpu_info()` → check GPUs before heavy runs.",
        "- `job_list(limit=20)` → review recent jobs and states.",
        "- `job_logs(job_id=\"...\")` → stream stdout/stderr; tee to paths.logs.",
        "- `tail_file(path=paths.logs + \"/run.log\", follow_ms=2000)` → live file tail.",
    ]
    sections["quick_commands"] = "\n".join(quick_lines)

    paths_lines = [
        "## Paths",
        f"- Standard log sink: `{log_sink_rel}` (see `paths.logs`).",
        f"- Shared notes root: `{notes_rel}` (append via `jsonl_append`).",
        f"- Job metadata: `{jobs_rel}` (one folder per job).",
    ]
    sections["paths"] = "\n".join(paths_lines)

    docs_lines = [
        "## Docs",
        "- Keep long-running tasks streaming into the log sink for consistency.",
        "- Capture issue summaries with `jsonl_append()` in the notes directory.",
        "- Use `list_dir()` and `stat()` before copying large artifacts.",
    ]
    sections["docs"] = "\n".join(docs_lines)

    if normalized_mode == "full":
        more = [
            "## Extended tips",
            "- `wait_for_path()` helps coordinate producers/consumers of artifacts.",
            "- `hash_file()` verifies downloads without leaving the jail.",
            "- `remove()` defaults to trashing; set `trash=false` for permanent deletes.",
        ]
        sections["extended"] = "\n".join(more)
        if "extended" not in requested_sections:
            requested_sections.append("extended")

    markdown_parts = [sections[name] for name in requested_sections if name in sections]
    markdown = "\n\n".join(markdown_parts)

    formatted_templates = [
        {
            "name": tpl["name"],
            "description": tpl["description"],
            "command": tpl["command"].format(logs=log_sink_rel),
        }
        for tpl in DEFAULT_JOB_TEMPLATES
    ]

    return {
        "markdown": markdown,
        "paths": {
            "logs": log_sink_rel,
            "notes": notes_rel,
            "jobs": jobs_rel,
        },
        "env": {
            "write_enabled": ENABLE_WRITE,
            "exec_enabled": ENABLE_EXEC,
            "fs_root": str(FS_ROOT),
            "log_sink_env": "MCP_LOG_SINK",
        },
        "templates": formatted_templates,
        "agents": _gather_agents_docs(),
    }

# ------------------------------- FS tooling ---------------------------------


@mcp.tool()
def list_dir(
    path: str = ".",
    depth: int = 1,
    glob: Optional[Sequence[str] | str] = None,
    include_hidden: bool = False,
    files_only: bool = False,
    dirs_only: bool = False,
    max_entries: int = 500,
    sort_by: str = "name",
    order: str = "asc",
    modified_since: Optional[str] = None,
) -> Any:
    """List directory entries with optional filters.
    Use to inspect artifacts quickly; tip: combine depth+glob to stay targeted.
    Usage: list_dir(path="results", depth=1, glob="*.jsonl", sort_by="mtime")
    Params: path:str, depth:int, glob:str|list[str], sort_by:"name"|"mtime"|"size", order:"asc"|"desc"
    Returns: [{path,name,is_dir,size,mtime,mime}]
    Gotchas: Huge trees with depth>2 can be slow without filters."""

    try:
        base = _resolve_path(path or ".")
    except ValueError as exc:
        return _error("EPERM", str(exc))

    if not base.exists():
        return _error("ENOENT", f"Path not found: {path}")
    if not base.is_dir():
        return _error("ENOTDIR", f"Not a directory: {path}")

    if glob is None:
        patterns: List[str] = []
    elif isinstance(glob, str):
        patterns = [glob]
    else:
        patterns = [str(pat) for pat in glob]

    def matches(rel: Path) -> bool:
        if not patterns:
            return True
        return any(rel.match(pat) for pat in patterns)

    cutoff_ts: Optional[float] = None
    if modified_since:
        try:
            cutoff_ts = datetime.fromisoformat(modified_since).timestamp()
        except ValueError:
            return _error("EINVAL", "modified_since must be ISO-8601 timestamp")

    results: List[Dict[str, Any]] = []
    max_depth = max(0, depth)
    if max_depth == 0:
        return []

    for root, dirs, files in os.walk(base):
        rootp = Path(root)
        rel_root = rootp.relative_to(base)
        rel_depth = len(rel_root.parts)

        if not dirs_only:
            for name in files:
                p = rootp / name
                if not include_hidden and p.name.startswith("."):
                    continue
                rel = p.relative_to(FS_ROOT)
                if not matches(rel):
                    continue
                md = _file_metadata(p)
                if cutoff_ts and md["mtime"] < cutoff_ts:
                    continue
                results.append(md)
                if len(results) >= max_entries:
                    break
            if len(results) >= max_entries:
                break

        if not files_only:
            for d in list(dirs):
                p = rootp / d
                if not include_hidden and p.name.startswith("."):
                    continue
                rel = p.relative_to(FS_ROOT)
                if not matches(rel):
                    continue
                md = _file_metadata(p)
                if cutoff_ts and md["mtime"] < cutoff_ts:
                    continue
                results.append(md)
                if len(results) >= max_entries:
                    break
            if len(results) >= max_entries:
                break

        if rel_depth + 1 >= max_depth:
            dirs[:] = []

    key_map = {
        "name": lambda item: item["name"],
        "mtime": lambda item: item["mtime"],
        "size": lambda item: item["size"],
    }
    key = key_map.get(sort_by, key_map["name"])
    reverse = order.lower() == "desc"
    results.sort(key=key, reverse=reverse)

    return results


@mcp.tool()
def stat(path: str) -> Dict[str, Any]:
    """Show metadata for one filesystem entry.
    Use before big reads to confirm size/time; tip: call init() for log sink discovery.
    Usage: stat(path="artifacts/log.txt")
    Params: path:str
    Returns: {path,name,is_dir,size,mtime,mime}
    Gotchas: Returns ENOENT if the path is missing."""

    try:
        p = _resolve_path(path)
    except ValueError as exc:
        return _error("EPERM", str(exc))

    if not p.exists():
        return _error("ENOENT", f"Path not found: {path}")
    return _file_metadata(p)


@mcp.tool()
def read_file(path: str, offset: int = 0, max_bytes: int = 256_000) -> Dict[str, Any]:
    """Read part of a file under FS_ROOT.
    Use offset/max_bytes to avoid huge dumps; tip: pair with stat().
    Usage: read_file(path="notes/run.jsonl", offset=0, max_bytes=200000)
    Params: path:str, offset:int, max_bytes:int
    Returns: {kind:"text"|"bytes", text|base64, truncated:bool}
    Gotchas: Binary files return base64."""

    try:
        p = _resolve_path(path)
    except ValueError as exc:
        return _error("EPERM", str(exc))
    if not p.exists() or not p.is_file():
        return _error("ENOENT", f"File not found: {path}")
    if offset < 0 or max_bytes <= 0:
        return _error("EINVAL", "offset must be >=0 and max_bytes >0")

    with p.open("rb") as f:
        if offset > 0:
            f.seek(offset)
        data = f.read(max_bytes + 1)
    sample = data[:4096]
    truncated = len(data) > max_bytes
    rel = str(p.relative_to(FS_ROOT))
    if is_probably_text(sample):
        try:
            text = data[:max_bytes].decode("utf-8", errors="replace")
        except Exception:
            text = data[:max_bytes].decode("latin-1", errors="replace")
        return {
            "kind": "text",
            "path": rel,
            "offset": offset,
            "text": text,
            "truncated": truncated,
        }
    return {
        "kind": "bytes",
        "path": rel,
        "offset": offset,
        "base64": base64.b64encode(data[:max_bytes]).decode("ascii"),
        "truncated": truncated,
    }


@mcp.tool()
def search(
    query: str,
    path: str = "",
    filename_only: bool = False,
    case_sensitive: bool = False,
    max_results: int = 100,
    max_file_size: Optional[int] = None,
    include_hidden: bool = False,
    regex: bool = False,
    glob: Optional[Sequence[str] | str] = None,
) -> Dict[str, Any]:
    """Search filenames and file content.
    Use glob to narrow scope; tip: switch regex=true for advanced patterns.
    Usage: search(query="error", path="logs", glob=["*.log"], regex=false)
    Params: query:str, path:str, glob:list[str], regex:bool, max_results:int,
      max_file_size:int|None
    Returns: {results:[{path,title,snippet}]}
    Gotchas: Large trees may take time."""

    patterns: List[str]
    if glob is None:
        patterns = []
    elif isinstance(glob, str):
        patterns = [glob]
    else:
        patterns = [str(pat) for pat in glob]

    start_time = time.perf_counter()
    search_root = path or "."
    configured_max_file_size = (
        max_file_size if max_file_size is not None else SEARCH_MAX_FILE_BYTES
    )
    if configured_max_file_size and configured_max_file_size > 0:
        read_limit = configured_max_file_size
    else:
        read_limit = None
    log_max_file_size = read_limit if read_limit is not None else "unlimited"
    logger.debug(
        "Search started query=%r path=%s filename_only=%s case_sensitive=%s max_results=%d max_file_size=%s include_hidden=%s regex=%s glob=%s",
        query,
        search_root,
        filename_only,
        case_sensitive,
        max_results,
        log_max_file_size,
        include_hidden,
        regex,
        patterns,
    )

    try:
        base = _resolve_path(search_root)
    except ValueError as exc:
        logger.warning("Search path rejected: %s", exc)
        elapsed = time.perf_counter() - start_time
        logger.info(
            "Search completed path=%s query=%r results=%d max_results=%d elapsed=%.3fs status=%s",
            search_root,
            query,
            0,
            max_results,
            elapsed,
            "error",
        )
        return _error("EPERM", str(exc))

    if not base.exists():
        elapsed = time.perf_counter() - start_time
        logger.info(
            "Search completed path=%s query=%r results=%d max_results=%d elapsed=%.3fs status=%s",
            search_root,
            query,
            0,
            max_results,
            elapsed,
            "missing",
        )
        return {"results": []}

    def matches(rel: Path) -> bool:
        if not patterns:
            return True
        return any(rel.match(pat) for pat in patterns)

    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        pat = re.compile(query if regex else re.escape(query), flags)
    except re.error as exc:
        elapsed = time.perf_counter() - start_time
        logger.warning("Search pattern rejected: %s", exc)
        logger.info(
            "Search completed path=%s query=%r results=%d max_results=%d elapsed=%.3fs status=%s",
            search_root,
            query,
            0,
            max_results,
            elapsed,
            "invalid_pattern",
        )
        return _error("EINVAL", f"Invalid pattern: {exc}")

    results: List[Dict[str, Any]] = []
    reached_limit = False
    for root, _, files in os.walk(base):
        rootp = Path(root)
        for name in files:
            p = rootp / name
            if not include_hidden and p.name.startswith("."):
                continue
            rel = p.relative_to(FS_ROOT)
            if not matches(rel):
                continue
            try:
                size = p.stat().st_size
            except OSError as exc:
                logger.debug("Skipping %s during search due to stat error: %s", rel, exc)
                continue
            if read_limit is not None and size > read_limit:
                logger.debug(
                    "Skipping %s during search: size %s exceeds limit %s",
                    rel,
                    size,
                    read_limit,
                )
                continue
            added = False
            if pat.search(name):
                md = _file_metadata(p)
                md["id"] = str(rel)
                md["title"] = p.name
                md["snippet"] = ""
                results.append(md)
                added = True
            if not added and not filename_only:
                read_result = _read_text_limited(p, read_limit)
                if read_result.error is not None:
                    logger.debug(
                        "Skipping %s during search due to read error: %s",
                        rel,
                        read_result.error,
                    )
                    continue
                if read_result.is_binary:
                    logger.debug("Skipping %s during search (binary detected)", rel)
                    continue
                text = read_result.text
                if text is None:
                    continue
                m = pat.search(text)
                if m:
                    start = max(0, m.start() - 80)
                    end = min(len(text), m.end() + 80)
                    snippet = text[start:end].strip()
                    md = _file_metadata(p)
                    md["id"] = str(rel)
                    md["title"] = p.name
                    md["snippet"] = snippet
                    results.append(md)
            if len(results) >= max_results:
                reached_limit = True
                break
        if reached_limit:
            break

    elapsed = time.perf_counter() - start_time
    logger.info(
        "Search completed path=%s query=%r results=%d max_results=%d elapsed=%.3fs status=%s",
        search_root,
        query,
        len(results),
        max_results,
        elapsed,
        "ok" if not reached_limit else "truncated",
    )
    return {"results": results}


@mcp.tool()
def fetch(id: str, max_bytes: int = 512_000, as_text: bool = True) -> Dict[str, Any]:
    """Fetch content/metadata for a discovered item.
    Use after list/search to pull a file; tip: set as_text=false for binaries.
    Usage: fetch(id="notes/todo.md", max_bytes=100000, as_text=true)
    Params: id:str, max_bytes:int, as_text:bool
    Returns: {id,path,mime,size,text|base64,truncated:bool}
    Gotchas: Requires id from prior discovery call."""

    try:
        p = _resolve_path(id)
    except ValueError as exc:
        return _error("EPERM", str(exc))
    if not p.exists() or not p.is_file():
        return _error("ENOENT", f"File not found: {id}")

    content = read_text_safely(p, max_bytes)
    mime, _ = mimetypes.guess_type(str(p))
    md = _file_metadata(p)
    public_base = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
    url = f"{public_base}/raw/{id}" if public_base else None
    out: Dict[str, Any] = {
        "id": id,
        "path": md["path"],
        "mime": mime or "application/octet-stream",
        "size": md["size"],
        "url": url,
        "truncated": content.get("truncated", False),
    }
    if content["kind"] == "text" and as_text:
        out["text"] = content["text"]
    else:
        out["base64"] = content.get("base64")
    return out


@mcp.tool()
def grep(
    pattern: str,
    path: str = ".",
    glob: Optional[str] = None,
    case_sensitive: bool = False,
    max_file_size: int = 2_000_000,
    limit: int = 200,
    include_hidden: bool = False,
    before: int = 0,
    after: int = 0,
) -> Dict[str, Any]:
    """Run regex search with context lines.
    Use when you need line numbers; tip: adjust before/after for more context.
    Usage: grep(pattern="ERROR", path="logs", before=2, after=2, limit=200)
    Params: pattern:str, path:str, before:int, after:int, limit:int
    Returns: {results:[{path,line_no,text}]}
    Gotchas: Skips non-text files automatically."""

    try:
        base = _resolve_path(path or ".")
    except ValueError as exc:
        return _error("EPERM", str(exc))

    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        pat = re.compile(pattern, flags)
    except re.error as exc:
        return _error("EINVAL", f"Invalid pattern: {exc}")

    out: Dict[str, Any] = {"results": []}
    if not base.exists():
        return out

    max_results = max(1, limit)

    for root, _, files in os.walk(base):
        rootp = Path(root)
        for name in files:
            p = rootp / name
            if not include_hidden and p.name.startswith("."):
                continue
            rel = p.relative_to(FS_ROOT)
            if glob and not rel.match(glob):
                continue
            try:
                if p.stat().st_size > max_file_size:
                    continue
                data = p.read_bytes()
            except Exception:
                continue
            if not is_probably_text(data[:4096]):
                continue
            try:
                text = data.decode("utf-8", errors="ignore")
            except Exception:
                text = data.decode("latin-1", errors="ignore")
            lines = text.splitlines()
            for m in pat.finditer(text):
                line_index = text.count("\n", 0, m.start())
                context_start = max(0, line_index - before)
                context_end = min(len(lines), line_index + after + 1)
                snippet = "\n".join(lines[context_start:context_end])
                out["results"].append(
                    {
                        "path": str(rel),
                        "line_no": line_index + 1,
                        "text": snippet,
                    }
                )
                if len(out["results"]) >= max_results:
                    return out
    return out


@mcp.tool()
def write_file(
    path: str,
    content: str = "",
    encoding: str = "utf-8",
    create_parents: bool = True,
    overwrite: bool = False,
    append: bool = False,
    atomic: bool = False,
    expected_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Write or append a file relative to FS_ROOT.
    Use for small/medium artifacts; tip: set atomic=true for config updates.
    Usage: write_file(path="configs/run.yaml", content="key: value", create_parents=true, atomic=true)
    Params: path:str, content:str, create_parents:bool, append:bool, atomic:bool, encoding:str
    Returns: {ok:true, bytes:int, created:bool, path:str}
    Gotchas: encoding="base64" treats content as encoded bytes."""

    if not ENABLE_WRITE:
        return _error("EPERM", "Write access is disabled")

    try:
        target = _resolve_path(path)
    except ValueError as exc:
        return _error("EPERM", str(exc))

    parent = target.parent
    if not parent.exists():
        if create_parents:
            try:
                _ensure_dir(parent)
            except Exception as exc:
                return _error("EPERM", f"Failed to create parent directories: {exc}")
        else:
            return _error("ENOENT", f"Parent directory does not exist: {parent}")

    existed = target.exists()
    if existed and target.is_dir():
        return _error("EISDIR", "Cannot write to a directory")

    if existed and not append and not overwrite:
        return _error("EEXIST", "File exists; set overwrite=true or append=true")

    if existed and expected_sha256:
        try:
            current_hash = _sha256_bytes(target.read_bytes())
        except Exception as exc:
            return _error("EIO", f"Failed to read existing file: {exc}")
        if current_hash != expected_sha256:
            return _error("EAGAIN", "Existing file hash mismatch", current_sha256=current_hash)

    try:
        if encoding.lower() == "base64":
            data = base64.b64decode(content.encode("ascii"), validate=True)
        else:
            data = content.encode(encoding)
    except Exception as exc:
        return _error("EINVAL", f"Unable to encode content: {exc}")

    created = not existed
    sha256: Optional[str] = None
    bytes_written = len(data)

    if append:
        if atomic:
            existing = b""
            if existed:
                try:
                    existing = target.read_bytes()
                except Exception as exc:
                    return _error("EIO", f"Failed to read existing file: {exc}")
            final_data = existing + data
            tmp = target.with_name(f"{target.name}.{uuid.uuid4().hex}.tmp")
            try:
                with tmp.open("wb") as f:
                    f.write(final_data)
                tmp.replace(target)
            finally:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
            sha256 = _sha256_bytes(final_data)
        else:
            try:
                with target.open("ab") as f:
                    f.write(data)
            except Exception as exc:
                return _error("EIO", f"Failed to append file: {exc}")
    else:
        tmp = None
        if atomic:
            tmp = target.with_name(f"{target.name}.{uuid.uuid4().hex}.tmp")
            try:
                with tmp.open("wb") as f:
                    f.write(data)
                tmp.replace(target)
            finally:
                if tmp and tmp.exists():
                    tmp.unlink(missing_ok=True)
        else:
            try:
                with target.open("wb") as f:
                    f.write(data)
            except Exception as exc:
                return _error("EIO", f"Failed to write file: {exc}")
        sha256 = _sha256_bytes(data)

    if sha256 is None:
        try:
            sha256 = _sha256_bytes(target.read_bytes())
        except Exception:
            sha256 = None

    return {
        "ok": True,
        "path": str(target.relative_to(FS_ROOT)),
        "bytes": bytes_written,
        "created": created,
        "sha256": sha256,
    }


@mcp.tool()
def make_dir(path: str, parents: bool = True) -> Dict[str, Any]:
    """Create a directory under FS_ROOT.
    Use before placing new assets; tip: leave parents=true for nested paths.
    Usage: make_dir(path="experiments/run-001", parents=true)
    Params: path:str, parents:bool
    Returns: {ok:true, path:str, created:bool}
    Gotchas: Requires write access."""

    if not ENABLE_WRITE:
        return _error("EPERM", "Write access is disabled")
    try:
        target = _resolve_path(path)
    except ValueError as exc:
        return _error("EPERM", str(exc))

    if target.exists():
        if target.is_dir():
            return {"ok": True, "path": str(target.relative_to(FS_ROOT)), "created": False}
        return _error("EEXIST", "A non-directory entry already exists at that path")

    if not parents and not target.parent.exists():
        return _error("ENOENT", "Parent directory does not exist")

    try:
        target.mkdir(parents=parents, exist_ok=True)
    except Exception as exc:
        return _error("EIO", f"Failed to create directory: {exc}")

    return {"ok": True, "path": str(target.relative_to(FS_ROOT)), "created": True}


@mcp.tool()
def remove(path: str, recursive: bool = False, trash: bool = True) -> Dict[str, Any]:
    """Remove a file or directory, optionally via trash.
    Use to clean artifacts; tip: default trash=true keeps a recovery copy.
    Usage: remove(path="experiments/tmp", recursive=true, trash=true)
    Params: path:str, recursive:bool, trash:bool
    Returns: {ok:true, path:str, trashed:bool, trash_path?:str}
    Gotchas: Set recursive=true for non-empty directories."""

    if not ENABLE_WRITE:
        return _error("EPERM", "Write access is disabled")

    try:
        target = _resolve_path(path)
    except ValueError as exc:
        return _error("EPERM", str(exc))

    if not target.exists():
        return _error("ENOENT", f"Path not found: {path}")

    if target.is_dir():
        has_children = any(target.iterdir())
        if has_children and not recursive:
            return _error("ENOTEMPTY", "Directory not empty; set recursive=true")
    trashed_path: Optional[Path] = None

    try:
        if trash:
            trash_root = _trash_root_path()
            _ensure_dir(trash_root)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            dest = trash_root / f"{timestamp}-{target.name}"
            counter = 0
            while dest.exists():
                counter += 1
                dest = trash_root / f"{timestamp}-{counter}-{target.name}"
            _ensure_dir(dest.parent)
            target.replace(dest)
            trashed_path = dest
        else:
            if target.is_dir():
                if recursive:
                    shutil.rmtree(target)
                else:
                    target.rmdir()
            else:
                target.unlink()
    except Exception as exc:
        return _error("EIO", f"Failed to remove path: {exc}")

    result: Dict[str, Any] = {
        "ok": True,
        "path": str(target.relative_to(FS_ROOT)),
        "trashed": bool(trashed_path),
    }
    if trashed_path:
        result["trash_path"] = str(trashed_path.relative_to(FS_ROOT))
    return result


@mcp.tool()
def move(src: str, dst: str, overwrite: bool = False, create_parents: bool = True) -> Dict[str, Any]:
    """Move or rename an entry inside the jail.
    Use to reorganize outputs; tip: enable overwrite for replacing targets.
    Usage: move(src="runs/a.log", dst="runs/archive/a.log", overwrite=false)
    Params: src:str, dst:str, overwrite:bool
    Returns: {ok:true, src:str, dst:str}
    Gotchas: Source and destination must stay inside FS_ROOT."""

    if not ENABLE_WRITE:
        return _error("EPERM", "Write access is disabled")

    try:
        src_path = _resolve_path(src)
        dst_path = _resolve_path(dst)
    except ValueError as exc:
        return _error("EPERM", str(exc))

    if not src_path.exists():
        return _error("ENOENT", f"Source not found: {src}")
    if src_path == dst_path:
        return {
            "ok": True,
            "src": str(src_path.relative_to(FS_ROOT)),
            "dst": str(dst_path.relative_to(FS_ROOT)),
        }

    if create_parents:
        try:
            _ensure_dir(dst_path.parent)
        except Exception as exc:
            return _error("EIO", f"Failed to create destination parent: {exc}")
    elif not dst_path.parent.exists():
        return _error("ENOENT", "Destination parent does not exist")

    if dst_path.exists():
        if not overwrite:
            return _error("EEXIST", "Destination already exists")
        try:
            if dst_path.is_dir():
                shutil.rmtree(dst_path)
            else:
                dst_path.unlink()
        except Exception as exc:
            return _error("EIO", f"Failed to clear destination: {exc}")

    try:
        shutil.move(str(src_path), str(dst_path))
    except Exception as exc:
        return _error("EIO", f"Failed to move entry: {exc}")

    return {
        "ok": True,
        "src": str(src_path.relative_to(FS_ROOT)),
        "dst": str(dst_path.relative_to(FS_ROOT)),
    }


@mcp.tool()
def copy(
    src: str,
    dst: str,
    overwrite: bool = False,
    preserve_attrs: bool = False,
    create_parents: bool = True,
) -> Dict[str, Any]:
    """Copy a file or directory within FS_ROOT.
    Use when branching experiments; tip: preserve_attrs=false avoids chmod surprises.
    Usage: copy(src="data/raw", dst="data/backup", overwrite=false, preserve_attrs=false)
    Params: src:str, dst:str, overwrite:bool, preserve_attrs:bool
    Returns: {ok:true, src:str, dst:str}
    Gotchas: Directories require overwrite=true when target exists."""

    if not ENABLE_WRITE:
        return _error("EPERM", "Write access is disabled")

    try:
        src_path = _resolve_path(src)
        dst_path = _resolve_path(dst)
    except ValueError as exc:
        return _error("EPERM", str(exc))

    if not src_path.exists():
        return _error("ENOENT", f"Source not found: {src}")

    if create_parents:
        try:
            _ensure_dir(dst_path.parent)
        except Exception as exc:
            return _error("EIO", f"Failed to create destination parent: {exc}")
    elif not dst_path.parent.exists():
        return _error("ENOENT", "Destination parent does not exist")

    if dst_path.exists():
        if not overwrite:
            return _error("EEXIST", "Destination already exists")
        try:
            if dst_path.is_dir():
                shutil.rmtree(dst_path)
            else:
                dst_path.unlink()
        except Exception as exc:
            return _error("EIO", f"Failed to clear destination: {exc}")

    try:
        if src_path.is_dir():
            shutil.copytree(
                src_path,
                dst_path,
                copy_function=shutil.copy2 if preserve_attrs else shutil.copy,
            )
        else:
            if preserve_attrs:
                shutil.copy2(src_path, dst_path)
            else:
                shutil.copy(src_path, dst_path)
    except Exception as exc:
        return _error("EIO", f"Failed to copy entry: {exc}")

    return {
        "ok": True,
        "src": str(src_path.relative_to(FS_ROOT)),
        "dst": str(dst_path.relative_to(FS_ROOT)),
    }


@mcp.tool()
def symlink(target: str, link_path: str, overwrite: bool = False) -> Dict[str, Any]:
    """Create a symlink pointing to another jailed path.
    Use for lightweight aliases; tip: keep targets inside FS_ROOT.
    Usage: symlink(target="data/raw", link_path="data/latest", overwrite=false)
    Params: target:str, link_path:str, overwrite:bool
    Returns: {ok:true, link_path:str, target:str}
    Gotchas: Rejects targets escaping the jail."""

    if not ENABLE_WRITE:
        return _error("EPERM", "Write access is disabled")

    try:
        link_abs = _resolve_path(link_path)
    except ValueError as exc:
        return _error("EPERM", str(exc))

    try:
        link_parent = link_abs.parent
        _ensure_dir(link_parent)
    except Exception as exc:
        return _error("EIO", f"Failed to prepare link parent: {exc}")

    target_path = Path(target)
    if target_path.is_absolute():
        resolved_target = target_path.resolve()
    else:
        resolved_target = (link_parent / target_path).resolve()

    if os.path.commonpath([str(resolved_target), str(FS_ROOT)]) != str(FS_ROOT):
        return _error("EPERM", "Symlink target escapes FS_ROOT")

    if link_abs.exists() or link_abs.is_symlink():
        if not overwrite:
            return _error("EEXIST", "Link path already exists")
        try:
            if link_abs.is_dir() and not link_abs.is_symlink():
                shutil.rmtree(link_abs)
            else:
                link_abs.unlink()
        except Exception as exc:
            return _error("EIO", f"Failed to clear existing link: {exc}")

    rel_target = os.path.relpath(resolved_target, start=link_parent)

    try:
        os.symlink(rel_target, link_abs)
    except Exception as exc:
        return _error("EIO", f"Failed to create symlink: {exc}")

    return {
        "ok": True,
        "link_path": str(link_abs.relative_to(FS_ROOT)),
        "target": rel_target,
    }


# ----------------------- tail/read helpers & utilities ----------------------


def _tail_offset(path: Path, lines: int) -> int:
    """Return the byte offset that starts the final ``lines`` of ``path``."""
    if lines <= 0:
        return 0

    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return 0

    if size == 0:
        return 0

    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        end = f.tell()
        block_size = 8192
        buffer = b""
        pos = end
        newline_count = 0

        while pos > 0 and newline_count <= lines:
            read_size = min(block_size, pos)
            pos -= read_size
            f.seek(pos)
            chunk = f.read(read_size)
            buffer = chunk + buffer
            newline_count += chunk.count(b"\n")
            if pos == 0:
                break

        endswith_newline = buffer.endswith(b"\n")
        total_lines = newline_count + (0 if endswith_newline else 1)
        if total_lines <= lines:
            return 0

        skip = total_lines - lines
        if skip <= 0:
            return 0

        seen = 0
        for idx, byte in enumerate(buffer):
            if byte == 10:  # "\n"
                seen += 1
                if seen == skip:
                    return pos + idx + 1

        return 0


def _squash_repeats(text: str, threshold: int = 3) -> str:
    """Collapse consecutive duplicate lines if they exceed ``threshold``."""

    if threshold <= 0:
        return text

    lines = text.splitlines(keepends=True)
    if not lines:
        return text

    parts: List[str] = []
    prev = None
    count = 0

    def flush(line: Optional[str], repetitions: int) -> None:
        if line is None or repetitions <= 0:
            return
        if repetitions <= threshold:
            parts.append(line * repetitions)
            return

        parts.append(line)
        if line.endswith("\n"):
            parts.append(f"[... repeated {repetitions - 1} more times ...]\n")
        else:
            parts.append(f"\n[... repeated {repetitions - 1} more times ...]")

    for line in lines:
        if line == prev:
            count += 1
        else:
            flush(prev, count)
            prev = line
            count = 1

    flush(prev, count)
    return "".join(parts)


@mcp.tool()
def tail_file(
    path: str,
    tail_lines: int = 200,
    follow_ms: int = 0,
    max_bytes: int = 200_000,
) -> Dict[str, Any]:
    """Tail a file with optional follow semantics.
    Use for log watching; tip: combine with init().paths.logs to monitor runs.
    Usage: tail_file(path="logs/run.log", tail_lines=200, follow_ms=2000, max_bytes=200000)
    Params: path:str, tail_lines:int, follow_ms:int, max_bytes:int
    Returns: {path, offset, next_offset, content:{kind,text|base64,truncated}}
    Gotchas: Binary files stream as base64."""

    try:
        target = _resolve_path(path)
    except ValueError as exc:
        return _error("EPERM", str(exc))

    if not target.exists() or not target.is_file():
        return _error("ENOENT", f"File not found: {path}")
    if max_bytes <= 0:
        return _error("EINVAL", "max_bytes must be positive")
    if tail_lines < 0:
        return _error("EINVAL", "tail_lines must be >=0")

    offset = _tail_offset(target, tail_lines) if tail_lines else 0
    deadline = time.time() + max(0, follow_ms) / 1000.0

    while True:
        size = target.stat().st_size
        if size > offset:
            with target.open("rb") as f:
                f.seek(offset)
                chunk = f.read(max_bytes)
            next_offset = offset + len(chunk)
            truncated = len(chunk) == max_bytes
            sample = chunk[:4096]
            rel = str(target.relative_to(FS_ROOT))
            if is_probably_text(sample):
                try:
                    text = chunk.decode("utf-8", errors="replace")
                except Exception:
                    text = chunk.decode("latin-1", errors="replace")
                content = {"kind": "text", "text": text, "truncated": truncated}
            else:
                content = {
                    "kind": "bytes",
                    "base64": base64.b64encode(chunk).decode("ascii"),
                    "truncated": truncated,
                }
            return {
                "path": rel,
                "offset": offset,
                "next_offset": next_offset,
                "content": content,
            }
        if follow_ms <= 0 or time.time() >= deadline:
            return {
                "path": str(target.relative_to(FS_ROOT)),
                "offset": offset,
                "next_offset": size,
                "content": {"kind": "text", "text": "", "truncated": False},
            }
        time.sleep(0.2)


@mcp.tool()
def wait_for_path(
    path: str,
    timeout_ms: int = 60_000,
    exist: bool = True,
    poll_ms: int = 200,
) -> Dict[str, Any]:
    """Wait until a path exists or disappears.
    Use to sync producers/consumers; tip: tighten timeout for quick checks.
    Usage: wait_for_path(path="models/latest.pt", timeout_ms=60000, exist=true)
    Params: path:str, timeout_ms:int, exist:bool
    Returns: {path:str, exists:bool, elapsed_ms:int}
    Gotchas: Returns ETIME on timeout."""

    try:
        target = _resolve_path(path)
    except ValueError as exc:
        return _error("EPERM", str(exc))

    start = time.time()
    deadline = start + max(0, timeout_ms) / 1000.0
    poll_interval = max(0.05, poll_ms / 1000.0)

    while time.time() <= deadline:
        exists = target.exists()
        if exists == exist:
            elapsed_ms = int((time.time() - start) * 1000)
            return {
                "path": str(target.relative_to(FS_ROOT)),
                "exists": exists,
                "elapsed_ms": elapsed_ms,
            }
        time.sleep(poll_interval)

    return _error("ETIME", "Timed out waiting for desired path state", path=str(target.relative_to(FS_ROOT)))


@mcp.tool()
def hash_file(path: str, algo: str = "sha256") -> Dict[str, Any]:
    """Compute a file hash for integrity checks.
    Use after downloads; tip: stick with sha256 unless compatibility demands md5.
    Usage: hash_file(path="data.bin", algo="sha256")
    Params: path:str, algo:"sha256"|"md5"
    Returns: {path:str, algo:str, hexdigest:str, size:int}
    Gotchas: Rejects directories with EISDIR."""

    try:
        target = _resolve_path(path)
    except ValueError as exc:
        return _error("EPERM", str(exc))

    if not target.exists():
        return _error("ENOENT", f"Path not found: {path}")
    if target.is_dir():
        return _error("EISDIR", "Cannot hash a directory")

    algo_norm = algo.lower()
    if algo_norm not in {"sha256", "md5"}:
        return _error("EINVAL", "Unsupported hash algorithm")

    hash_func = hashlib.sha256 if algo_norm == "sha256" else hashlib.md5
    h = hash_func()
    size = 0
    try:
        with target.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                size += len(chunk)
                h.update(chunk)
    except Exception as exc:
        return _error("EIO", f"Failed to read file: {exc}")

    return {
        "path": str(target.relative_to(FS_ROOT)),
        "algo": algo_norm,
        "hexdigest": h.hexdigest(),
        "size": size,
    }


@mcp.tool()
def jsonl_append(path: str, record: Dict[str, Any], create_parents: bool = True) -> Dict[str, Any]:
    """Append one structured record to a JSONL file (atomic).
    Use to log issues/attempts/findings without losing context; tip: keep tags consistent.
    Usage: jsonl_append(path="notes/issues.jsonl", record={
  "who":"agent", "goal":"run X", "expected":"success", "observed":"timeout",
  "error":"ConnectionError", "attempts":["restart","smaller batch"], "next":"try Y",
  "tags":["infra","timeout"], "artifacts":["jobs/123/log"]
}, create_parents=true)
    Params: path:str, record:dict, create_parents:bool
    Returns: {ok:true, bytes_written:int}
    Gotchas: Large files are rewritten per append for atomicity."""

    if not ENABLE_WRITE:
        return _error("EPERM", "Write access is disabled")
    if not isinstance(record, dict):
        return _error("EINVAL", "record must be a JSON-serializable object")

    try:
        bytes_written = _append_jsonl_atomic(path, record, create_parents=create_parents)
    except ValueError as exc:
        return _error("EPERM", str(exc))
    except Exception as exc:
        return _error("EIO", f"Failed to append record: {exc}")

    return {"ok": True, "bytes_written": bytes_written}


# --------------------------- Background job manager ---------------------------

JOBS: Dict[str, Dict[str, Any]] = {}


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    tmp.replace(path)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _default_allow() -> List[str]:
    return [
        "python",
        "python3",
        "pip",
        "uv",
        "pytest",
        "bash",
        "git",
        "nvidia-smi",
        "ls",
        "cat",
        "head",
        "tail",
    ]


def _jobs_root() -> Path:
    d = FS_ROOT / ".mcp_jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _job_dir(job_id: str) -> Path:
    d = _jobs_root() / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _job_meta_path(job_id: str) -> Path:
    return _job_dir(job_id) / "job.json"


def _job_log_path(job_id: str) -> Path:
    return _job_dir(job_id) / "stdout_stderr.log"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso_from_timestamp(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


@mcp.tool()
def job_start(
    cmd: List[str] | str,
    cwd: str = ".",
    env: Optional[Dict[str, str]] = None,
    shell: bool = False,
    name: Optional[str] = None,
    allowlist: Optional[List[str]] = None,
    timeout_s: Optional[int] = None,
) -> Dict[str, Any]:
    """Start a detached process whose stdout/stderr stream to a log file.
    Use for long jobs; tip: All long-running tasks should emit to the standard log sink and monitor via job_logs or tail_file.
    Usage: job_start(cmd=["bash","-lc","python train.py | tee \"$LOG_SINK/train.log\""])
    Params: cmd:list[str]|str, cwd:str, env:dict[str,str], shell:bool, timeout_s:int
    Returns: {job_id,pid,log_path,log_url}
    Gotchas: Requires MCP_ENABLE_EXEC=1."""

    if not ENABLE_EXEC:
        return _error("EPERM", "Execution is disabled")

    workdir: Path
    try:
        workdir = _resolve_path(cwd or ".")
    except ValueError as exc:
        return _error("EPERM", str(exc))

    if isinstance(cmd, str) and not shell:
        cmd = shlex.split(cmd)
    if not cmd:
        return _error("EINVAL", "Command must not be empty")

    allow = set(allowlist or _default_allow())
    exe = "bash" if shell else (cmd if isinstance(cmd, str) else cmd[0])
    base_exe = os.path.basename(exe if isinstance(exe, str) else str(exe))
    if base_exe not in allow:
        return _error("EPERM", "Denied by allowlist", exe=base_exe, allow=sorted(allow))

    job_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    log_path = _job_log_path(job_id)

    env_final = os.environ.copy()
    if env:
        for k, v in env.items():
            if not isinstance(k, str) or not isinstance(v, str):
                return _error("EINVAL", "env must be a string-to-string mapping")
            env_final[k] = v
    env_final.setdefault("LOG_SINK", str(_log_sink_path()))

    with open(log_path, "ab", buffering=0) as lf:
        lf.write(f"[{_now_iso()}] job_start {job_id} name={name or ''} cmd={cmd}\n".encode("utf-8"))
    try:
        proc = subprocess.Popen(
            cmd if not shell else " ".join(cmd if isinstance(cmd, list) else [cmd]),
            cwd=str(workdir),
            env=env_final,
            shell=shell,
            stdout=open(log_path, "ab", buffering=0),
            stderr=open(log_path, "ab", buffering=0),
            start_new_session=True,
        )
    except Exception as exc:
        return _error("EIO", f"Failed to launch process: {exc}")

    timeout_deadline = None
    if timeout_s is not None:
        if not isinstance(timeout_s, int) or timeout_s < 0:
            return _error("EINVAL", "timeout_s must be a non-negative integer")
        if timeout_s > 0:
            timeout_deadline = time.time() + timeout_s

    meta = {
        "id": job_id,
        "pid": proc.pid,
        "name": name,
        "cmd": cmd if isinstance(cmd, list) else [cmd],
        "cwd": str(workdir.relative_to(FS_ROOT)),
        "shell": shell,
        "allowlist": sorted(list(allow)),
        "started_at": _now_iso(),
        "finished_at": None,
        "returncode": None,
        "log_path": str(log_path.relative_to(FS_ROOT)),
        "log_sink": str(_log_sink_path().relative_to(FS_ROOT)),
    }

    if timeout_deadline is not None:
        meta["timeout_s"] = timeout_s
        meta["timeout_at"] = _iso_from_timestamp(timeout_deadline)
        meta["timed_out"] = False

    _write_json(_job_meta_path(job_id), meta)
    JOBS[job_id] = {"popen": proc, "meta": meta, "log_path": str(log_path)}

    if timeout_deadline is not None:
        def _auto_timeout():
            remaining = timeout_deadline - time.time()
            if remaining > 0:
                time.sleep(remaining)
            if proc.poll() is not None:
                return
            try:
                with open(log_path, "ab", buffering=0) as lf:
                    lf.write(
                        f"[{_now_iso()}] job_timeout {job_id} after {timeout_s}s\n".encode("utf-8")
                    )
            except Exception:
                pass
            job_stop(job_id, wait_ms=0)
            latest_meta = _read_json(_job_meta_path(job_id)) or meta.copy()
            latest_meta["timed_out"] = True
            latest_meta.setdefault("finished_at", _now_iso())
            _write_json(_job_meta_path(job_id), latest_meta)

        threading.Thread(target=_auto_timeout, name=f"job_timeout_{job_id}", daemon=True).start()

    public_base = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
    log_url = f"{public_base}/raw/{meta['log_path']}" if public_base else None
    return {"job_id": job_id, "pid": proc.pid, "log_path": meta["log_path"], "log_url": log_url}


@mcp.tool()
def job_status(job_id: str) -> Dict[str, Any]:
    """Return the latest known status for a background job.
    Use to poll progress; tip: inspect state for running/succeeded/failed.
    Usage: job_status(job_id="1234-abcd")
    Params: job_id:str
    Returns: {job_id,state,returncode,running,started_at,finished_at}
    Gotchas: Job IDs expire if metadata is removed."""

    meta = _read_json(_job_meta_path(job_id))
    if not meta:
        return _error("ENOENT", f"Unknown job: {job_id}")

    proc = JOBS.get(job_id, {}).get("popen")
    if proc is not None:
        rc = proc.poll()
        if rc is None:
            alive = True
        else:
            alive = False
            meta["finished_at"] = meta.get("finished_at") or _now_iso()
            meta["returncode"] = rc
            _write_json(_job_meta_path(job_id), meta)
            JOBS.get(job_id, {}).pop("popen", None)
    else:
        alive = False

    rc = meta.get("returncode")
    if alive:
        state = "running"
    elif rc is None:
        state = "unknown"
    elif rc == 0:
        state = "succeeded"
    else:
        state = "failed"

    meta["state"] = state
    meta["running"] = alive
    return meta


@mcp.tool()
def job_list(limit: int = 50) -> Dict[str, Any]:
    """List recent jobs with status summaries.
    Use to review activity; tip: pass limit to avoid giant outputs.
    Usage: job_list(limit=50)
    Params: limit:int
    Returns: {jobs:[{job_id,state,started_at,description}]}
    Gotchas: Includes jobs still running."""

    jobs = []
    for d in sorted(_jobs_root().iterdir(), key=lambda p: p.name, reverse=True):
        if not d.is_dir():
            continue
        meta = _read_json(d / "job.json")
        if not meta:
            continue
        status = job_status(meta.get("id"))
        if "error" in status:
            continue
        cmd = status.get("cmd") or []
        description = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
        jobs.append(
            {
                "job_id": status.get("id"),
                "state": status.get("state"),
                "started_at": status.get("started_at"),
                "description": description.strip(),
            }
        )
        if len(jobs) >= limit:
            break
    return {"jobs": jobs}


@mcp.tool()
def job_logs(
    job_id: str,
    offset: int = 0,
    max_bytes: int = 200_000,
    follow_ms: int = 0,
    tail_lines: Optional[int] = None,
    squash_repeats: bool = False,
) -> Dict[str, Any]:
    """Fetch or follow logs for a job.
    Use for debugging and progress; tip: tee long runs into the standard log sink.
    Usage: job_logs(job_id="1234", tail_lines=200, follow_ms=2000, squash_repeats=true)
    Params: job_id:str, tail_lines:int, follow_ms:int, max_bytes:int, squash_repeats:bool
    Returns: {job_id,offset,next_offset,running,content:{kind,text,truncated}}
    Gotchas: Returns ENOENT if the job log vanished."""

    logp = _job_log_path(job_id)
    if not logp.exists():
        return _error("ENOENT", f"No log file for job {job_id}")

    if max_bytes <= 0:
        return _error("EINVAL", "max_bytes must be positive")

    if tail_lines is None and offset < 0:
        tail_lines = abs(offset)
        offset = 0

    if tail_lines is not None:
        if not isinstance(tail_lines, int):
            return _error("EINVAL", "tail_lines must be an integer")
        if tail_lines > 0:
            offset = _tail_offset(logp, tail_lines)

    wait_deadline = time.time() + max(max(follow_ms, 0), 0) / 1000.0
    next_offset = offset
    while True:
        size = logp.stat().st_size
        if size > offset:
            with logp.open("rb") as f:
                f.seek(offset)
                data = f.read(max_bytes)
            next_offset = offset + len(data)
            text = data.decode("utf-8", errors="replace")
            if squash_repeats:
                text = _squash_repeats(text)
            content = {
                "kind": "text",
                "text": text,
                "truncated": len(data) == max_bytes,
            }
            st = job_status(job_id)
            running = st.get("running", False) if "error" not in st else False
            return {
                "job_id": job_id,
                "offset": offset,
                "next_offset": next_offset,
                "running": running,
                "content": content,
            }
        if time.time() >= wait_deadline or follow_ms <= 0:
            st = job_status(job_id)
            running = st.get("running", False) if "error" not in st else False
            return {
                "job_id": job_id,
                "offset": offset,
                "next_offset": size,
                "running": running,
                "content": {"kind": "text", "text": "", "truncated": False},
            }
        time.sleep(0.2)


@mcp.tool()
def job_stop(job_id: str, sig: int = 15, kill_children: bool = True, wait_ms: int = 5000) -> Dict[str, Any]:
    """Send a signal to a background job.
    Use to stop runaway tasks; tip: leave kill_children=true to clean process groups.
    Usage: job_stop(job_id="1234", sig=15, kill_children=true, wait_ms=5000)
    Params: job_id:str, sig:int, kill_children:bool, wait_ms:int
    Returns: {job_id,stopped:bool,killed:bool?}
    Gotchas: Requires the job to have been started by this server."""

    meta = _read_json(_job_meta_path(job_id))
    if not meta:
        return _error("ENOENT", f"Unknown job: {job_id}")
    pid = int(meta.get("pid", -1))
    if pid <= 0:
        return _error("EINVAL", "Job has no valid PID")

    def _signal(sig_to_send: int) -> bool:
        try:
            if kill_children:
                os.killpg(pid, sig_to_send)
            else:
                os.kill(pid, sig_to_send)
            return True
        except ProcessLookupError:
            return False
        except Exception:
            return False

    if not _signal(sig):
        return {"job_id": job_id, "stopped": False, "missing": True}

    deadline = time.time() + max(0, wait_ms) / 1000.0
    while _pid_alive(pid) and time.time() < deadline:
        time.sleep(0.1)
    if _pid_alive(pid):
        _signal(signal.SIGKILL)
        return {"job_id": job_id, "killed": True}
    else:
        meta["finished_at"] = meta.get("finished_at") or _now_iso()
        _write_json(_job_meta_path(job_id), meta)
        return {"job_id": job_id, "stopped": True}


@mcp.tool()
def gpu_info() -> Dict[str, Any]:
    """Show GPU availability and memory summary.
    Use before heavy runs; tip: rerun if queueing jobs for live status.
    Usage: gpu_info()
    Returns: {available:bool, gpus:[{name,total_mem_mb,free_mem_mb}]}
    Gotchas: Returns error details if nvidia-smi is unavailable."""

    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=5)
    except FileNotFoundError:
        return {"available": False, "gpus": [], "error": _error("ENOENT", "nvidia-smi not found")["error"]}
    except subprocess.TimeoutExpired:
        return {"available": False, "gpus": [], "error": _error("ETIME", "nvidia-smi timed out")["error"]}
    except Exception as exc:
        return {"available": False, "gpus": [], "error": _error("EIO", f"nvidia-smi failed: {exc}")["error"]}

    if proc.returncode != 0:
        err = (proc.stderr or "").strip()
        return {
            "available": False,
            "gpus": [],
            "error": _error("EIO", err or f"nvidia-smi exit {proc.returncode}")["error"],
        }

    lines = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
    gpus = []

    def _as_int(val: str) -> Optional[int]:
        try:
            return int(float(val))
        except (TypeError, ValueError):
            return None

    for idx, line in enumerate(lines):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            gpus.append({"index": idx, "raw": line})
            continue
        index_raw, name, mem_total, mem_used, mem_free = parts
        try:
            index = int(index_raw)
        except ValueError:
            index = index_raw
        gpus.append(
            {
                "index": index,
                "name": name,
                "memory_total_mb": _as_int(mem_total),
                "memory_used_mb": _as_int(mem_used),
                "memory_free_mb": _as_int(mem_free),
            }
        )

    return {"available": bool(gpus), "gpus": gpus}


@mcp.tool()
def request_additional_resources(
    summary: str,
    details: Optional[str] = None,
    urgency: str = "normal",
    contact: Optional[str] = None,
) -> Dict[str, Any]:
    """Log a request for extra resources or permissions.
    Use when blocked; tip: add clear urgency so admins can triage.
    Usage: request_additional_resources(summary="Need dataset access", urgency="high")
    Params: summary:str, details:str, urgency:str, contact:str
    Returns: {status:"recorded", request_id:str, log_path:str}
    Gotchas: Writes into the shared notes root."""

    summary_clean = (summary or "").strip()
    if not summary_clean:
        return _error("EINVAL", "summary is required")

    details_clean = details.strip() if details and details.strip() else None
    urgency_raw = (urgency or "normal").strip()
    urgency_norm = urgency_raw.lower()
    allowed = {"low", "normal", "high", "critical"}
    if urgency_norm not in allowed:
        urgency_norm = "normal"

    entry: Dict[str, Any] = {
        "id": uuid.uuid4().hex,
        "submitted_at": _now_iso(),
        "summary": summary_clean,
        "urgency": urgency_norm,
    }
    if details_clean:
        entry["details"] = details_clean
    if urgency_raw and urgency_raw.lower() != urgency_norm:
        entry["reported_urgency"] = urgency_raw
    if contact and contact.strip():
        entry["contact"] = contact.strip()

    try:
        log_path = _append_jsonl_atomic("hello-mcp/notes/resource_requests.jsonl", entry, create_parents=True)
    except Exception as exc:
        return _error("EIO", f"failed to record request: {exc}")

    return {
        "status": "recorded",
        "request_id": entry["id"],
        "log_path": "hello-mcp/notes/resource_requests.jsonl",
        "request": entry,
    }


@mcp.tool()
def admin_feedback(
    message: str,
    topic: str = "general",
    allow_follow_up: Any = False,
    contact: Optional[str] = None,
) -> Dict[str, Any]:
    """Send actionable feedback to admins.
    Use for bugs or UX gaps; tip: include repro steps for faster follow-up.
    Usage: admin_feedback(message="write_file rejects JSON payloads", tags=["fs","bug"])
    Params: message:str, topic:str, allow_follow_up:bool, contact:str
    Returns: {status:"recorded", feedback_id:str, log_path:str}
    Gotchas: Stored in notes; avoid sensitive data."""

    message_clean = (message or "").strip()
    if not message_clean:
        return _error("EINVAL", "message is required")

    topic_clean = (topic or "general").strip() or "general"

    entry: Dict[str, Any] = {
        "id": uuid.uuid4().hex,
        "submitted_at": _now_iso(),
        "topic": topic_clean,
        "message": message_clean,
        "allow_follow_up": _normalize_bool(allow_follow_up),
    }

    if contact and contact.strip():
        entry["contact"] = contact.strip()

    try:
        _append_jsonl_atomic("hello-mcp/notes/admin_feedback.jsonl", entry, create_parents=True)
    except Exception as exc:
        return _error("EIO", f"failed to record feedback: {exc}")

    return {
        "status": "recorded",
        "feedback_id": entry["id"],
        "log_path": "hello-mcp/notes/admin_feedback.jsonl",
        "feedback": entry,
    }


# --------------------------- ASGI app & gateway -----------------------------


async def health(_req):
    return PlainTextResponse("ok")


async def raw(request):
    rel_path = request.path_params["path"]
    try:
        p = _resolve_path(rel_path)
    except ValueError:
        return JSONResponse({"error": "not_found"}, status_code=404)
    if not p.exists() or not p.is_file():
        return JSONResponse({"error": "not_found"}, status_code=404)
    return FileResponse(str(p), filename=p.name)


class MCPGateway:
    """ASGI wrapper to keep /mcp stable while FastMCP expects root mounts."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path = scope.get("path", "")
            method = scope.get("method", "GET").upper()
            if path in ("/mcp", "/mcp/") and method in {"GET", "HEAD", "OPTIONS"}:
                resp = Response(status_code=204)
                return await resp(scope, receive, send)
            if path == "/mcp" and method in {"POST", "OPTIONS"}:
                scope = dict(scope)
                scope["path"] = "/mcp/"
                return await self.app(scope, receive, send)
        return await self.app(scope, receive, send)


def build_app() -> Starlette:
    @contextlib.asynccontextmanager
    async def lifespan(app):
        async with mcp.session_manager.run():
            yield

    mcp.settings.streamable_http_path = "/"
    mcp_asgi = mcp.streamable_http_app()

    app = Starlette(
        routes=[
            Route("/health", endpoint=health),
            Route("/raw/{path:path}", endpoint=raw),
            Mount("/mcp", app=mcp_asgi),
        ],
        lifespan=lifespan,
    )
    return MCPGateway(app)


# ---------------------------------- main ------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Run the MCP filesystem server.")
    parser.add_argument("--root", default=os.environ.get("FS_ROOT", "."), help="Root directory (jail).")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    args = parser.parse_args()

    global FS_ROOT
    FS_ROOT = resolve_base(args.root)

    app = build_app()
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()
