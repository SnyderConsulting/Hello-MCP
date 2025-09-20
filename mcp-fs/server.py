#!/usr/bin/env python3
# Read-only filesystem MCP server (Streamable HTTP) for RunPod.

import argparse
import base64
import contextlib
import json
import mimetypes
import os
import re
import signal
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from starlette.routing import Mount, Route
import uvicorn

import hashlib, shlex, subprocess

ENABLE_WRITE = os.environ.get("MCP_ENABLE_WRITE", "0") == "1"
ENABLE_EXEC  = os.environ.get("MCP_ENABLE_EXEC",  "0") == "1"

# ------------------------ FS helpers (read-only jail) ------------------------

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

def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256(); h.update(b); return h.hexdigest()

# ----------------------------- MCP server setup -----------------------------

mcp = FastMCP(
    name="RunPod Read-Only FS",
    instructions=(
        "Provides read-only browsing of a file tree inside a RunPod instance. "
        "All paths are resolved under FS_ROOT; path escapes are blocked. "
        "Use `search` to find files and `fetch` to retrieve file content. "
        "Extra tools: list_dir, stat, read_file."
    ),
)

FS_ROOT: Path = Path(".")

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

# ------------------------------- Tools --------------------------------------

@mcp.tool()
def list_dir(
    path: str = ".",
    depth: int = 1,
    glob: Optional[str] = None,
    include_hidden: bool = False,
    files_only: bool = False,
    dirs_only: bool = False,
    max_entries: int = 500,
) -> List[Dict[str, Any]]:
    base = safe_join(FS_ROOT, path)
    results: List[Dict[str, Any]] = []
    if not base.exists():
        return results

    def include(p: Path) -> bool:
        if not include_hidden and p.name.startswith("."): return False
        if files_only and not p.is_file(): return False
        if dirs_only and not p.is_dir(): return False
        return True

    depth = max(0, depth)
    if depth == 0:
        return []

    for root, dirs, files in os.walk(base):
        rootp = Path(root)
        rel_depth = len(rootp.relative_to(base).parts)

        if not dirs_only:
            for name in files:
                p = rootp / name
                if not include(p): continue
                if glob and not Path(str(p.relative_to(FS_ROOT))).match(glob): continue
                results.append(_file_metadata(p))
                if len(results) >= max_entries: return results

        if not files_only:
            for d in list(dirs):
                p = rootp / d
                if not include(p): continue
                if glob and not Path(str(p.relative_to(FS_ROOT))).match(glob): continue
                results.append(_file_metadata(p))
                if len(results) >= max_entries: return results

        if rel_depth + 1 >= depth:
            dirs[:] = []
    return results

@mcp.tool()
def stat(path: str) -> Dict[str, Any]:
    p = safe_join(FS_ROOT, path)
    if not p.exists():
        return {"error": "not_found"}
    return _file_metadata(p)

@mcp.tool()
def read_file(path: str, offset: int = 0, max_bytes: int = 256_000) -> Dict[str, Any]:
    p = safe_join(FS_ROOT, path)
    if not p.exists() or not p.is_file():
        return {"error": "not_found"}
    with p.open("rb") as f:
        if offset > 0: f.seek(max(0, offset))
        data = f.read(max_bytes + 1)
    sample = data[:4096]
    if is_probably_text(sample):
        try:
            text = data[:max_bytes].decode("utf-8", errors="replace")
        except Exception:
            text = data[:max_bytes].decode("latin-1", errors="replace")
        return {"kind": "text", "path": str(p.relative_to(FS_ROOT)), "offset": offset, "text": text, "truncated": len(data) > max_bytes}
    else:
        return {"kind": "bytes", "path": str(p.relative_to(FS_ROOT)), "offset": offset, "base64": base64.b64encode(data[:max_bytes]).decode("ascii"), "truncated": len(data) > max_bytes}

@mcp.tool()
def search(
    query: str,
    path: str = "",
    filename_only: bool = False,
    case_sensitive: bool = False,
    max_results: int = 100,
    include_hidden: bool = False,
) -> Dict[str, Any]:
    base = safe_join(FS_ROOT, path or ".")
    if not base.exists():
        return {"results": []}
    results: List[Dict[str, Any]] = []
    flags = 0 if case_sensitive else re.IGNORECASE
    pat = re.compile(re.escape(query), flags)

    def include(p: Path) -> bool:
        return include_hidden or not p.name.startswith(".")

    for root, _, files in os.walk(base):
        rootp = Path(root)
        for name in files:
            p = rootp / name
            if not include(p): continue
            rel = p.relative_to(FS_ROOT)
            added = False
            if pat.search(name):
                md = _file_metadata(p); md["id"] = str(rel); md["title"] = p.name; md["snippet"] = ""; results.append(md); added = True
            if not added and not filename_only:
                try:
                    with p.open("rb") as f: data = f.read(256_000)
                    if is_probably_text(data[:4096]):
                        try: text = data.decode("utf-8", errors="ignore")
                        except Exception: text = data.decode("latin-1", errors="ignore")
                        m = pat.search(text)
                        if m:
                            start = max(0, m.start() - 80); end = min(len(text), m.end() + 80)
                            snippet = text[start:end].strip()
                            md = _file_metadata(p); md["id"] = str(rel); md["title"] = p.name; md["snippet"] = snippet
                            results.append(md)
                except Exception:
                    pass
            if len(results) >= max_results:
                return {"results": results}
    return {"results": results}

@mcp.tool()
def fetch(id: str, max_bytes: int = 512_000, as_text: bool = True) -> Dict[str, Any]:
    p = safe_join(FS_ROOT, id)
    if not p.exists() or not p.is_file():
        return {"error": "not_found", "id": id}
    content = read_text_safely(p, max_bytes)
    mime, _ = mimetypes.guess_type(str(p))
    md = _file_metadata(p)
    public_base = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
    url = f"{public_base}/raw/{id}" if public_base else None
    out: Dict[str, Any] = {"id": id, "path": md["path"], "mime": mime or "application/octet-stream", "size": md["size"], "url": url, "truncated": content.get("truncated", False)}
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
    max_file_size: int = 2_000_000,   # skip huge files
    max_matches: int = 200,
    include_hidden: bool = False,
) -> Dict[str, Any]:
    """
    Simple text grep inside FS_ROOT. Returns up to max_matches occurrences with
    file, line, and a short snippet around the match.
    """
    base = safe_join(FS_ROOT, path or ".")
    flags = 0 if case_sensitive else re.IGNORECASE
    pat = re.compile(pattern, flags)
    out = {"results": []}
    if not base.exists():
        return out

    for root, _, files in os.walk(base):
        rootp = Path(root)
        for name in files:
            p = rootp / name
            if not include_hidden and p.name.startswith("."):
                continue
            rel = p.relative_to(FS_ROOT)
            if glob and not Path(str(rel)).match(glob):
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
            for m in pat.finditer(text):
                line_start = text.rfind("\n", 0, m.start()) + 1
                line_end   = text.find("\n", m.end())
                if line_end == -1: line_end = len(text)
                line_text = text[line_start:line_end]
                out["results"].append({
                    "path": str(rel),
                    "line": line_text[:500],
                    "start": m.start() - line_start,
                    "end":   m.end()   - line_start,
                })
                if len(out["results"]) >= max_matches:
                    return out
    return out


@mcp.tool()
def write_file(
    path: str,
    text: Optional[str] = None,
    base64_bytes: Optional[str] = None,
    encoding: str = "utf-8",
    overwrite: bool = False,
    expected_sha256: Optional[str] = None,
    create_parents: bool = True,
) -> Dict[str, Any]:
    """
    Write a file inside FS_ROOT (jail-scoped).
    - Provide either `text` or `base64_bytes`.
    - If `expected_sha256` is set and file exists with a different hash, abort.
    - If `overwrite=False` and file exists, abort.
    """
    if not ENABLE_WRITE:
        return {"error": "write_disabled"}

    if (text is None) == (base64_bytes is None):
        return {"error": "provide exactly one of: text or base64_bytes"}

    p = safe_join(FS_ROOT, path)
    if p.exists():
        current = p.read_bytes()
        cur_hash = _sha256_bytes(current)
        if expected_sha256 and cur_hash != expected_sha256:
            return {"error": "hash_mismatch", "current_sha256": cur_hash}
        if not overwrite:
            return {"error": "exists"}

    data: bytes
    if text is not None:
        data = text.encode(encoding)
    else:
        try:
            data = base64.b64decode(base64_bytes, validate=True)
        except Exception as e:
            return {"error": f"invalid_base64: {e}"}

    if create_parents:
        p.parent.mkdir(parents=True, exist_ok=True)

    p.write_bytes(data)
    return {
        "path": str(p.relative_to(FS_ROOT)),
        "size": len(data),
        "sha256": _sha256_bytes(data),
        "overwrote": p.exists(),
    }


@mcp.tool()
def run(
    cmd: List[str] | str,
    cwd: str = ".",
    timeout_s: int = 30,
    max_output: int = 200_000,
    env: Optional[Dict[str, str]] = None,
    allowlist: Optional[List[str]] = None,
    shell: bool = False,
) -> Dict[str, Any]:
    """
    Execute a bounded command under FS_ROOT.
    - jail-scoped cwd
    - timeout and output caps
    - simple allowlist for the executable (default safe set)
    NOTE: enable with MCP_ENABLE_EXEC=1
    """
    if not ENABLE_EXEC:
        return {"error": "exec_disabled"}

    workdir = safe_join(FS_ROOT, cwd or ".")
    if isinstance(cmd, str) and not shell:
        cmd = shlex.split(cmd)
    if not cmd:
        return {"error": "empty_command"}

    allow = set(allowlist or _default_allow())
    exe = cmd if isinstance(cmd, str) else cmd[0]
    if shell:
        exe = "bash"
    base_exe = os.path.basename(exe)
    if base_exe not in allow:
        return {"error": "denied_by_allowlist", "exe": base_exe, "allow": sorted(allow)}

    # Build environment
    env_final = os.environ.copy()
    if env:
        for k, v in (env or {}).items():
            if not isinstance(k, str) or not isinstance(v, str):  # keep it simple
                return {"error": "env_must_be_str_kv"}
            env_final[k] = v

    start = time.time()
    try:
        proc = subprocess.run(
            cmd if not shell else " ".join(cmd if isinstance(cmd, list) else [cmd]),
            cwd=str(workdir),
            env=env_final,
            shell=shell,
            capture_output=True,
            timeout=timeout_s,
        )
        rc = proc.returncode
        out = proc.stdout or b""
        err = proc.stderr or b""
    except subprocess.TimeoutExpired as e:
        rc = -1
        out = (e.stdout or b"") + b"\n[TIMEOUT]"
        err = (e.stderr or b"") + b""
    except Exception as e:
        return {"error": f"exec_failed: {e}"}

    def cap_and_decode(b: bytes) -> Dict[str, Any]:
        truncated = len(b) > max_output
        b = b[:max_output]
        if is_probably_text(b[:4096]):
            try:
                t = b.decode("utf-8", errors="replace")
            except Exception:
                t = b.decode("latin-1", errors="replace")
            return {"kind": "text", "text": t, "truncated": truncated}
        return {"kind": "bytes", "base64": base64.b64encode(b).decode("ascii"), "truncated": truncated}

    elapsed = time.time() - start
    return {
        "returncode": rc,
        "stdout": cap_and_decode(out),
        "stderr": cap_and_decode(err),
        "elapsed_s": round(elapsed, 3),
        "cwd": str(workdir.relative_to(FS_ROOT)),
        "cmd": cmd if isinstance(cmd, list) else [cmd],
    }

# ----------------------- Background job manager (no timeouts) -----------------------

JOBS: Dict[str, Dict[str, Any]] = {}  # in-memory (augments on-disk metadata)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        return True  # can't signal but likely exists


def _default_allow() -> List[str]:
    return ["python", "pip", "uv", "pytest", "bash", "git", "nvidia-smi", "ls", "cat", "head", "tail"]


@mcp.tool()
def job_start(
    cmd: List[str] | str,
    cwd: str = ".",
    env: Optional[Dict[str, str]] = None,
    shell: bool = False,
    name: Optional[str] = None,
    allowlist: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Start a detached process whose stdout/stderr are tee'd to a log file.
    Returns {job_id, pid, log_path, log_url}.
    """

    workdir = safe_join(FS_ROOT, cwd or ".")
    if isinstance(cmd, str) and not shell:
        cmd = shlex.split(cmd)
    if not cmd:
        return {"error": "empty_command"}

    # allowlist gate
    allow = set(allowlist or _default_allow())
    exe = "bash" if shell else (cmd if isinstance(cmd, str) else cmd[0])
    base_exe = os.path.basename(exe if isinstance(exe, str) else str(exe))
    if base_exe not in allow:
        return {"error": "denied_by_allowlist", "exe": base_exe, "allow": sorted(allow)}

    job_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    log_path = _job_log_path(job_id)

    # Build env
    env_final = os.environ.copy()
    if env:
        for k, v in env.items():
            if not isinstance(k, str) or not isinstance(v, str):
                return {"error": "env_must_be_str_kv"}
            env_final[k] = v

    # Open log file in append; create new process group (for group kill)
    with open(log_path, "ab", buffering=0) as lf:
        lf.write(f"[{_now_iso()}] job_start {job_id} name={name or ''} cmd={cmd}\n".encode("utf-8"))
    proc = subprocess.Popen(
        cmd if not shell else " ".join(cmd if isinstance(cmd, list) else [cmd]),
        cwd=str(workdir),
        env=env_final,
        shell=shell,
        stdout=open(log_path, "ab", buffering=0),
        stderr=open(log_path, "ab", buffering=0),
        start_new_session=True,  # new process group, so we can kill children
    )

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
    }
    _write_json(_job_meta_path(job_id), meta)
    JOBS[job_id] = {"popen": proc, "meta": meta, "log_path": str(log_path)}

    public_base = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
    log_url = f"{public_base}/raw/{meta['log_path']}" if public_base else None
    return {"job_id": job_id, "pid": proc.pid, "log_path": meta["log_path"], "log_url": log_url}


@mcp.tool()
def job_status(job_id: str) -> Dict[str, Any]:
    meta = _read_json(_job_meta_path(job_id))
    if not meta:
        return {"error": "not_found", "job_id": job_id}

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
    else:
        alive = _pid_alive(meta.get("pid", -1))
        # returncode unknown unless we’re the parent; leave as-is

    return {
        "job_id": job_id,
        "running": alive,
        "pid": meta.get("pid"),
        "returncode": meta.get("returncode"),
        "started_at": meta.get("started_at"),
        "finished_at": meta.get("finished_at"),
        "cwd": meta.get("cwd"),
        "name": meta.get("name"),
        "log_path": meta.get("log_path"),
    }


@mcp.tool()
def job_list(limit: int = 50) -> Dict[str, Any]:
    root = _jobs_root()
    jobs = []
    for d in sorted(root.iterdir(), key=lambda p: p.name, reverse=True):
        if not d.is_dir():
            continue
        meta = _read_json(d / "job.json")
        if meta:
            jobs.append(
                {
                    "job_id": meta.get("id"),
                    "name": meta.get("name"),
                    "started_at": meta.get("started_at"),
                    "returncode": meta.get("returncode"),
                }
            )
        if len(jobs) >= limit:
            break
    return {"jobs": jobs}


@mcp.tool()
def job_logs(job_id: str, offset: int = 0, max_bytes: int = 200_000, follow_ms: int = 0) -> Dict[str, Any]:
    """
    Return a chunk of the job log starting at byte 'offset', and the next_offset.
    If follow_ms > 0, wait up to that many milliseconds for new data (max ~25s).
    """

    logp = _job_log_path(job_id)
    if not logp.exists():
        return {"error": "not_found"}
    wait_deadline = time.time() + min(max(follow_ms, 0), 25_000) / 1000.0
    next_offset = offset
    while True:
        size = logp.stat().st_size
        if size > offset:
            with logp.open("rb") as f:
                f.seek(offset)
                data = f.read(max_bytes)
            next_offset = offset + len(data)
            content = {
                "kind": "text",
                "text": data.decode("utf-8", errors="replace"),
                "truncated": len(data) == max_bytes,
            }
            st = job_status(job_id)  # returns current running state
            return {
                "job_id": job_id,
                "offset": offset,
                "next_offset": next_offset,
                "running": st.get("running", False),
                "content": content,
            }
        if time.time() >= wait_deadline or follow_ms <= 0:
            st = job_status(job_id)
            return {
                "job_id": job_id,
                "offset": offset,
                "next_offset": size,
                "running": st.get("running", False),
                "content": {"kind": "text", "text": "", "truncated": False},
            }
        time.sleep(0.2)


@mcp.tool()
def job_stop(job_id: str, sig: int = 15, kill_children: bool = True, wait_ms: int = 5000) -> Dict[str, Any]:
    """
    Send a signal to the job (default TERM). If kill_children=True, signal the process group.
    Escalate to SIGKILL after wait_ms if still running.
    """

    meta = _read_json(_job_meta_path(job_id))
    if not meta:
        return {"error": "not_found", "job_id": job_id}
    pid = int(meta.get("pid", -1))
    if pid <= 0:
        return {"error": "bad_pid"}

    def _signal(sig_to_send: int):
        try:
            if kill_children:
                os.killpg(pid, sig_to_send)
            else:
                os.kill(pid, sig_to_send)
            return True
        except ProcessLookupError:
            return False
        except Exception as e:
            return {"error": f"signal_failed: {e}"}

    _signal(sig)
    deadline = time.time() + max(0, wait_ms) / 1000.0
    while _pid_alive(pid) and time.time() < deadline:
        time.sleep(0.1)
    if _pid_alive(pid):
        _signal(signal.SIGKILL)
        return {"job_id": job_id, "killed": True}
    else:
        meta["finished_at"] = meta.get("finished_at") or _now_iso()
        # we can’t reliably get returncode unless we were the parent; leave as-is
        _write_json(_job_meta_path(job_id), meta)
        return {"job_id": job_id, "stopped": True}

# --------------------------- ASGI app & gateway -----------------------------

async def health(_req):
    return PlainTextResponse("ok")

async def raw(request):
    rel_path = request.path_params["path"]
    p = safe_join(FS_ROOT, rel_path)
    if not p.exists() or not p.is_file():
        return JSONResponse({"error": "not_found"}, status_code=404)
    return FileResponse(str(p), filename=p.name)

class MCPGateway:
    """
    ASGI gateway:
      - GET/HEAD/OPTIONS to /mcp or /mcp/ -> 204 (no redirects)
      - POST/OPTIONS to /mcp -> rewritten to /mcp/ (to avoid Mount 307)
      - Everything else -> pass-through
    """
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
                scope = dict(scope); scope["path"] = "/mcp/"
                return await self.app(scope, receive, send)
        return await self.app(scope, receive, send)

def build_app() -> Starlette:
    # Start the MCP session manager when the root app starts
    @contextlib.asynccontextmanager
    async def lifespan(app):
        async with mcp.session_manager.run():
            yield

    # Configure MCP so its internal entrypoint path is '/'
    # When mounted at '/mcp', final public endpoint is exactly '/mcp' (and '/mcp/').
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
    parser = argparse.ArgumentParser(description="Run a read-only FS MCP server.")
    parser.add_argument("--root", default=os.environ.get("FS_ROOT", "."), help="Root directory (read-only).")
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
        proxy_headers=True,         # trust proxy headers (host/scheme) behind RunPod/CF
        forwarded_allow_ips="*",    # trust all proxies (OK for demo)
    )

if __name__ == "__main__":
    main()
