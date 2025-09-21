# Hello MCP (Stable Baseline)

This directory is the FS_ROOT exposed by your MCP server. It includes a minimal file-bridge and smoke tests.

## Structure
- mcp-bridge/ — helpers for reliable file transfers (ops.sh, inbox/, outbox/, uploads/tmp/)
- scripts/ — local helper scripts (e.g., smoke.sh)
- hello-mcp/notes/ — shared knowledge base and log area

## Quick start (server)
Prerequisites:

- Python 3.9+ with the `venv` module available (on Debian/Ubuntu install via `apt install python3 python3-venv`).

The helper script will reuse the active virtualenv if you already have one selected. Otherwise it will create `.venv/` in the repo root on first run and install the server requirements automatically.

Run from the host (outside the jail). Ensure port 8000 is exposed in the environment where the server runs so the proxy URL above can reach it. Then run:

```bash
# from the repo root (defaults auto-detect this checkout)
./mcp-fs/run_server.sh --restart \
  --public-url "https://<POD_ID>-8000.proxy.runpod.net" \
  --enable-write --enable-exec --foreground
```

The `--enable-exec` flag enables the job tools that launch commands inside the jail.

## Smoke test (inside MCP via job tools)
- Exec check: call `job_start` with `cmd=["bash", "-lc", "echo OK"]` and confirm output via `job_logs`.
- Bridge check: start `job_start` with `cmd=["bash", "-lc", "mcp-bridge/ops.sh self-test"]`.

## Notes
- Exec allow-list for job execution can be extended via env: `MCP_EXEC_ALLOW="bash sh python python3 pip uv pytest git nvidia-smi ls cat head tail ops.sh"`.
- If your client hides Exec until a tool is marked destructive, annotate `job_start` and `write_file` in server.py with `destructiveHint` (and `openWorldHint` for job_start).
- `job_start` accepts an optional `timeout_s` to auto-stop runaway processes, and `job_logs` understands `tail_lines`/negative `offset` plus `squash_repeats` to make tailing easier.
- Use the new `gpu_info` tool to return a lightweight summary from `nvidia-smi` (when available).
- To enable verbose tracing for filesystem search operations, start the server with `MCP_FS_LOG_LEVEL=DEBUG` (or configure Python logging) to see start/completion timing and skip details in the logs.
