#!/usr/bin/env bash
# Copyright 2026 The Kubernetes Authors.
# SPDX-License-Identifier: Apache-2.0
set -o errexit
set -o nounset
set -o pipefail

formal_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$formal_root"
SSA_LAKE="${SSA_LAKE:-lake}"
lean_binary="$("${SSA_LAKE}" env which lean)"
probe="$formal_root/.lake/source-mutation"
mkdir -p "$probe/SSA"
cp -R "$formal_root/.lake/build/lib/lean/SSA/." "$probe/SSA/"

# A semantic source mutation must invalidate the unchanged universal theorem.
python3 - ../../../../../../../../vendor/sigs.k8s.io/structured-merge-diff/v6/merge/update.go "$probe/update.go" <<'PY'
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text()
original = "mergedSet.Difference(prunedSet).Intersection(last)"
replacement = "mergedSet.Difference(prunedSet).Difference(last)"
if source.count(original) != 1:
    raise SystemExit("Cannot locate the unique source mutation; update verify-source.sh.")
Path(sys.argv[2]).write_text(source.replace(original, replacement))
PY
go run ./cmd/ssa-source -source "$probe/update.go" >"$probe/SSA/GoSource.lean"
cp SSA/SourceContracts.lean "$probe/SSA/SourceContracts.lean"
rm -f "$probe/SSA/GoSource.olean" "$probe/SSA/GoSource.olean.private" "$probe/SSA/GoSource.olean.server"
(
  cd "$probe"
  export LEAN_PATH="$probe:$formal_root/.lake/build/lib/lean"
  "$lean_binary" -DwarningAsError=true -o SSA/GoSource.olean SSA/GoSource.lean
  if "$lean_binary" -DwarningAsError=true SSA/SourceContracts.lean >proof-rejection.log 2>&1; then
    printf 'Source mutation unexpectedly satisfied the original theorem; see %s.\n' "$probe" >&2
    exit 1
  fi
  if ! grep -Eq 'unsolved goals|Tactic.*failed' proof-rejection.log; then
    cat proof-rejection.log >&2
    printf 'Source mutation failed for an unrelated reason; see %s.\n' "$probe" >&2
    exit 1
  fi
)
printf 'The unchanged Lean theorem correctly rejected the mutated Go source.\n'
