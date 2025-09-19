#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
ops.sh assemble <upload_id> <target_path> [--sha256 HEX]
ops.sh clean    <upload_id>
ops.sh self-test
USAGE
}

ROOT="mcp-bridge"
TMP="$ROOT/uploads/tmp"

assemble() {
  local upid=$1; shift
  local target=$1; shift
  local want_sha=""
  if [[ "${1-}" == "--sha256" ]]; then
    want_sha=$2; shift 2
  fi
  local dir="$TMP/$upid"
  [[ -d $dir ]] || { echo "no such upload: $upid"; exit 2; }
  mkdir -p "$(dirname "$target")"
  # concatenate in numeric order
  (cd "$dir" && ls -1 part.* | sort -V | xargs -I{} cat "{}") > "$target"
  local got_sha
  if command -v sha256sum >/dev/null 2>&1; then
    got_sha="$(sha256sum "$target" | awk '{print $1}')"
  else
    got_sha=""
  fi
  if [[ -n $want_sha && -n $got_sha && "$want_sha" != "$got_sha" ]]; then
    echo "sha256 mismatch: want=$want_sha got=$got_sha"
    exit 3
  fi
  echo "assembled: $target size=$(stat -c %s "$target") sha256=$got_sha"
}

clean() {
  local upid=$1
  rm -rf "$TMP/$upid"
  echo "cleaned: $TMP/$upid"
}

self_test() {
  local upid="upl_$(date +%s)"
  local dir="$TMP/$upid"
  mkdir -p "$dir"
  printf 'hello '  > "$dir/part.00000"
  printf 'world'   > "$dir/part.00001"
  assemble "$upid" "$ROOT/outbox/self_test.txt"
  grep -q "hello world" "$ROOT/outbox/self_test.txt"
  echo ok
}

cmd=${1-}
case "$cmd" in
  assemble) shift; assemble "$@";;
  clean)    shift; clean "$@";;
  self-test) shift; self_test;;
  *) usage; exit 1;;
esac
