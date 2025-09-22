#!/usr/bin/env bash
# run_server.sh — one-shot setup/runner for your RunPod MCP server
# Usage examples:
#   bash run_server.sh
#   bash run_server.sh --fs-root "$(pwd)/hello-mcp" --public-url https://<POD_ID>-8000.proxy.runpod.net --enable-write
#   bash run_server.sh --foreground   # run in the foreground (good for debugging)
#   bash run_server.sh --restart      # stop any old server and start fresh

set -Eeuo pipefail

# ---------- defaults ----------
# Resolve repository locations based on this script's directory so the
# defaults work no matter where the repo is cloned.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"

APP_DIR="${APP_DIR:-$SCRIPT_DIR}"
SERVER="${SERVER:-$APP_DIR/server.py}"
VENV="${VENV:-$REPO_ROOT/.venv}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
FS_ROOT="${FS_ROOT:-$REPO_ROOT/hello-mcp}"   # change to your repo root if you like
PUBLIC_BASE_URL="${PUBLIC_BASE_URL:-}"       # e.g. https://<POD_ID>-8000.proxy.runpod.net
MCP_ENABLE_WRITE="${MCP_ENABLE_WRITE:-0}"    # set 1 to allow write_file
MCP_ENABLE_EXEC="${MCP_ENABLE_EXEC:-0}"      # set 1 to allow job tools (job_start, etc.)
LOG="${LOG:-$APP_DIR/server.log}"
PIDFILE="${PIDFILE:-$APP_DIR/server.pid}"
PYTHON_CMD="${PYTHON_CMD:-}"

UPDATE_DEPS=1
FOREGROUND=0
RESTART=0

# ---------- args ----------
usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --fs-root PATH          Jail root for tools (default: $FS_ROOT)
  --port N                Port to bind (default: $PORT)
  --host IP               Host to bind (default: $HOST)
  --public-url URL        Public base URL (adds clickable links in fetch())
  --enable-write          Enable write_file tool (default off)
  --enable-exec           Enable job tools (default off)
  --no-update             Skip pip install/upgrade
  --server PATH           Path to server.py (default: $SERVER)
  --foreground            Run in foreground (logs to stdout)
  --restart               Kill old server then start fresh
  --log PATH              Log file (background mode)
  -h, --help              Show this help
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fs-root) FS_ROOT="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --host) HOST="$2"; shift 2;;
    --public-url) PUBLIC_BASE_URL="$2"; shift 2;;
    --enable-write) MCP_ENABLE_WRITE=1; shift;;
    --enable-exec) MCP_ENABLE_EXEC=1; shift;;
    --no-update) UPDATE_DEPS=0; shift;;
    --server) SERVER="$2"; shift 2;;
    --foreground) FOREGROUND=1; shift;;
    --restart) RESTART=1; shift;;
    --log) LOG="$2"; shift 2;;
    -h|--help) usage;;
    *) echo "Unknown arg: $1"; usage;;
  esac
done

# ---------- helpers ----------
die() { echo "✖ $*" >&2; exit 1; }
ok()  { echo "✔ $*"; }
info(){ echo "→ $*"; }

ensure_dirs() {
  mkdir -p "$APP_DIR"
  [[ -d "$FS_ROOT" ]] || die "FS_ROOT not found: $FS_ROOT"
  [[ -f "$SERVER" ]] || die "server.py not found at: $SERVER"
}

MIN_PY_MAJOR=3
MIN_PY_MINOR=10

_python_version_string() {
  local cmd="$1"
  "$cmd" -c 'import sys; print(".".join(str(p) for p in sys.version_info[:3]))'
}

_python_meets_min_version() {
  local cmd="$1"
  "$cmd" -c "import sys; sys.exit(0 if sys.version_info >= (${MIN_PY_MAJOR}, ${MIN_PY_MINOR}) else 1)"
}

resolve_python() {
  local candidate
  local first_old_cmd=""
  local first_old_version=""

  if [[ -n "${PYTHON_CMD:-}" ]]; then
    if ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
      die "Specified PYTHON_CMD not found: $PYTHON_CMD"
    fi
    if ! _python_meets_min_version "$PYTHON_CMD" >/dev/null 2>&1; then
      local version
      version="$(_python_version_string "$PYTHON_CMD" 2>/dev/null || true)"
      [[ -n "$version" ]] || version="unknown"
      die "Python at $PYTHON_CMD is version $version but the MCP server requires >= ${MIN_PY_MAJOR}.${MIN_PY_MINOR}."
    fi
    return 0
  fi

  for candidate in python3.12 python3.11 python3.10 python3 python; do
    if ! command -v "$candidate" >/dev/null 2>&1; then
      continue
    fi
    if _python_meets_min_version "$candidate" >/dev/null 2>&1; then
      PYTHON_CMD="$candidate"
      return 0
    fi
    if [[ -z "$first_old_cmd" ]]; then
      first_old_cmd="$candidate"
      first_old_version="$(_python_version_string "$candidate" 2>/dev/null || true)"
      [[ -n "$first_old_version" ]] || first_old_version="unknown"
    fi
  done

  if [[ -n "$first_old_cmd" ]]; then
    die "Found Python $first_old_version at $first_old_cmd, but the MCP server requires >= ${MIN_PY_MAJOR}.${MIN_PY_MINOR}. Install a newer Python or set PYTHON_CMD."
  fi

  die "python ${MIN_PY_MAJOR}.${MIN_PY_MINOR}+ not found. Install Python ${MIN_PY_MAJOR}.${MIN_PY_MINOR}+ (for example via pyenv) and rerun."
}

make_venv() {
  if [[ "$VENV" == "$REPO_ROOT/.venv" && -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" && ! -d "$VENV" ]]; then
    info "Using already active virtualenv at $VIRTUAL_ENV"
    VENV="$VIRTUAL_ENV"
  fi

  if [[ ! -x "$VENV/bin/python" ]]; then
    resolve_python
    info "Creating venv at $VENV"
    "$PYTHON_CMD" -m venv "$VENV" || die "Failed to create virtualenv at $VENV (ensure python3-venv is installed)."
  fi

  local activate="$VENV/bin/activate"
  if [[ ! -f "$activate" ]]; then
    die "Virtualenv activation script missing at $activate. Delete $VENV and rerun this script."
  fi

  # shellcheck disable=SC1090
  source "$activate"

  if ! _python_meets_min_version "$VENV/bin/python" >/dev/null 2>&1; then
    local current_version
    current_version="$(_python_version_string "$VENV/bin/python" 2>/dev/null || true)"
    [[ -n "$current_version" ]] || current_version="unknown"
    die "Virtualenv Python $current_version is too old; delete $VENV and rerun this script with Python ${MIN_PY_MAJOR}.${MIN_PY_MINOR}+ available."
  fi

  if (( UPDATE_DEPS )); then
    info "Installing/Updating Python deps…"
    "$VENV/bin/python" -m pip install -U pip >/dev/null
    if [[ -f "$REPO_ROOT/requirements.txt" ]]; then
      "$VENV/bin/python" -m pip install -U -r "$REPO_ROOT/requirements.txt" >/dev/null
    else
      "$VENV/bin/python" -m pip install -U "mcp[cli]" "uvicorn>=0.30" "starlette>=0.38" >/dev/null
    fi
    "$VENV/bin/python" -m pip install -r "$REPO_ROOT/projects/BAGEL/requirements.txt" >/dev/null 2>&1 || true
  fi

  local uvicorn_version="not installed"
  if "$VENV/bin/python" -c "import uvicorn" >/dev/null 2>&1; then
    uvicorn_version="$("$VENV/bin/python" -c 'import uvicorn; print(uvicorn.__version__)')"
  fi
  ok "Python: $("$VENV/bin/python" -V), Uvicorn: $uvicorn_version"
}

stop_old() {
  if [[ -f "$PIDFILE" ]]; then
    local pid
    pid="$(cat "$PIDFILE" || true)"
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      info "Stopping old server (pid $pid)…"
      kill "$pid" || true
      sleep 1
    fi
    rm -f "$PIDFILE"
  fi
  # Fallback: kill by path if lingering
  pkill -f "$SERVER" 2>/dev/null || true
}

health_check() {
  local url="http://127.0.0.1:$PORT/health"
  if command -v curl >/dev/null 2>&1; then
    for i in {1..40}; do
      if curl -fsS "$url" >/dev/null 2>&1; then return 0; fi
      sleep 0.25
    done
    return 1
  else
    # best-effort python check
    "$VENV/bin/python" - <<PY >/dev/null 2>&1 && return 0 || return 1
import urllib.request, sys
import time
url="$url"
for _ in range(40):
    try:
        with urllib.request.urlopen(url, timeout=1) as r:
            sys.exit(0 if r.status==200 else 1)
    except Exception:
        time.sleep(0.25)
sys.exit(1)
PY
  fi
}

start_bg() {
  info "Starting MCP server in background… (log: $LOG)"
  # shellcheck disable=SC2086
  env FS_ROOT="$FS_ROOT" PUBLIC_BASE_URL="$PUBLIC_BASE_URL" \
      MCP_ENABLE_WRITE="$MCP_ENABLE_WRITE" MCP_ENABLE_EXEC="$MCP_ENABLE_EXEC" \
      "$VENV/bin/python" "$SERVER" --root "$FS_ROOT" --host "$HOST" --port "$PORT" \
      >>"$LOG" 2>&1 &
  echo $! > "$PIDFILE"

  if health_check; then
    ok "Server healthy on http://127.0.0.1:$PORT/health"
  else
    echo "⚠ server may still be starting — tailing last lines:"
    tail -n 50 "$LOG" || true
  fi
}

start_fg() {
  info "Starting MCP server in foreground…"
  exec env FS_ROOT="$FS_ROOT" PUBLIC_BASE_URL="$PUBLIC_BASE_URL" \
      MCP_ENABLE_WRITE="$MCP_ENABLE_WRITE" MCP_ENABLE_EXEC="$MCP_ENABLE_EXEC" \
      "$VENV/bin/python" "$SERVER" --root "$FS_ROOT" --host "$HOST" --port "$PORT"
}

print_summary() {
  echo
  ok "MCP server running."
  echo "FS_ROOT          : $FS_ROOT"
  echo "Host/Port        : $HOST:$PORT"
  echo "Write enabled    : $MCP_ENABLE_WRITE"
  echo "Exec enabled     : $MCP_ENABLE_EXEC (job tools)"
  [[ -n "$PUBLIC_BASE_URL" ]] && echo "Public base URL   : $PUBLIC_BASE_URL"
  echo
  echo "Proxy URL (RunPod): https://<POD_ID>-$PORT.proxy.runpod.net"
  echo "MCP endpoint      : https://<POD_ID>-$PORT.proxy.runpod.net/mcp"
  echo
  echo "Manual probe (JSON-RPC list tools):"
  echo "curl -sS -X POST \"https://<POD_ID>-$PORT.proxy.runpod.net/mcp\" \\"
  echo "  -H 'accept: application/json, text/event-stream' \\"
  echo "  -H 'content-type: application/json' \\"
  echo "  --data '{\"jsonrpc\":\"2.0\",\"id\":\"1\",\"method\":\"tools/list\",\"params\":{}}' | jq ."
  echo
}

# ---------- main ----------
ensure_dirs
make_venv

if (( RESTART )); then stop_old; fi

if (( FOREGROUND )); then
  start_fg
else
  start_bg
  print_summary
fi

