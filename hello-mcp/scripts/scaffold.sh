#!/usr/bin/env bash
set -euo pipefail
DEST=${1:-my-mcp-project}
mkdir -p "$DEST"
# core pieces
cp -r mcp-bridge "$DEST/"
mkdir -p "$DEST/scripts"
cp scripts/smoke.sh "$DEST/scripts/"
# docs
cp README.md "$DEST/" 2>/dev/null || true
# git hygiene
cp .gitignore "$DEST/.gitignore" 2>/dev/null || true
# reset transient dirs
rm -rf "$DEST/mcp-bridge/uploads/tmp" "$DEST/mcp-bridge/outbox" "$DEST/mcp-bridge/inbox"
mkdir -p "$DEST/mcp-bridge/uploads/tmp" "$DEST/mcp-bridge/outbox" "$DEST/mcp-bridge/inbox"
echo "Scaffold created at $DEST"
