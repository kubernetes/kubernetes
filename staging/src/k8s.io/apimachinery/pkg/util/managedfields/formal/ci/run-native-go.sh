#!/usr/bin/env bash
# Copyright 2026 The Kubernetes Authors.
# SPDX-License-Identifier: Apache-2.0
set -o errexit
set -o nounset
set -o pipefail

if [[ "$#" -lt 2 ]]; then
  printf 'Usage: run-native-go.sh OVERLAY PACKAGE [go test arguments]\n' >&2
  exit 2
fi
formal_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
overlay="$1"
package="$2"
shift 2
native_go="${SSA_NATIVE_GO:-go}"
package_dir="$("$native_go" list -f '{{.Dir}}' "$package")"
cache_dir="$("$native_go" env GOMODCACHE)"
case "$package_dir/" in
  "$cache_dir/"*) ;;
  *) exec "$native_go" test -overlay="$overlay" "$package" "$@" ;;
esac

# Go forbids cache overlays; an isolated source copy leaves the installed dependency untouched.
module_dir="$("$native_go" list -f '{{.Module.Dir}}' "$package")"
selected_go="$("$native_go" env GOROOT)/bin/go"
native_tree="$(mktemp -d "$formal_root/.lake/native-module.XXXXXX")"
trap 'rm -rf "$native_tree"' EXIT
python3 - "$module_dir" "$overlay" "$native_tree" <<'PY'
import json
import os
import shutil
import stat
import sys
from pathlib import Path

source, overlay, scratch = map(Path, sys.argv[1:])
source = source.resolve()
copy = scratch / "module"
shutil.copytree(source, copy)
# Cache directories are read-only; only the new copy needs writable directories for cleanup.
for current, directories, files in os.walk(copy):
    current = Path(current)
    current.chmod(stat.S_IMODE(current.stat().st_mode) | stat.S_IWUSR)
replacements = {}
for target, replacement in json.loads(overlay.read_text())["Replace"].items():
    target = Path(target).resolve()
    try:
        target = copy / target.relative_to(source)
    except ValueError:
        pass
    replacements[str(target)] = str(Path(replacement).resolve()) if replacement else ""
(scratch / "overlay.json").write_text(json.dumps({"Replace": replacements}))
PY
(
  cd "$native_tree/module"
  GOWORK=off GOTOOLCHAIN=local "$selected_go" test -mod=readonly -overlay="$native_tree/overlay.json" "$package" "$@"
)
