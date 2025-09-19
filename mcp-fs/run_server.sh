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
MCP_ENABLE_EXEC="${MCP_ENABLE_EXEC:-0}"      # set 1 to allow run
LOG="${LOG:-$APP_DIR/server.log}"
PIDFILE="${PIDFILE:-$APP_DIR/server.pid}"

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
  --enable-exec           Enable run tool (default off)
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

make_venv() {
  if [[ ! -x "$VENV/bin/python" ]]; then
    info "Creating venv at $VENV"
    python3 -m venv "$VENV"
  fi
  # shellcheck disable=SC1090
  source "$VENV/bin/activate"
  if (( UPDATE_DEPS )); then
    info "Installing/Updating Python deps…"
    pip install -U pip >/dev/null
    pip install -U "mcp[cli]" "uvicorn>=0.30" "starlette>=0.38" >/dev/null
  fi
  ok "Python: $("$VENV/bin/python" -V), Uvicorn: $(python -c 'import uvicorn, sys; print(uvicorn.__version__)')"
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
  echo "Exec enabled     : $MCP_ENABLE_EXEC"
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

