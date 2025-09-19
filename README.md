# Hello MCP (Stable Baseline)

This directory is the FS_ROOT exposed by your MCP server. It includes a minimal file-bridge and smoke tests.

## Structure
- mcp-bridge/ — helpers for reliable file transfers (ops.sh, inbox/, outbox/, uploads/tmp/)
- scripts/ — local helper scripts (e.g., smoke.sh)
- notes/ — scratchpad

## Quick start (server)
Run from the host (outside the jail):

```bash
/workspace/mcp-fs/run_server.sh --restart \
  --fs-root /workspace/hello-mcp \
  --public-url "https://<POD_ID>-8000.proxy.runpod.net" \
  --enable-write --enable-exec --foreground
```

## Smoke test (inside MCP via run tool)
- Exec check: run `bash -lc "echo OK"`
- Bridge check: `bash -lc "mcp-bridge/ops.sh self-test"`

## Notes
- Exec allow-list can be extended via env: `MCP_EXEC_ALLOW="bash sh python pip uv pytest git nvidia-smi ls cat head tail ops.sh"`.
- If your client hides Exec until a tool is marked destructive, annotate `run` and `write_file` in server.py with `destructiveHint` (and `openWorldHint` for run).
