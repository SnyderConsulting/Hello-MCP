#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
chmod +x mcp-bridge/ops.sh
mcp-bridge/ops.sh self-test
