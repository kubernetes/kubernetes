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
updater_source="$(go list -f '{{.Dir}}' sigs.k8s.io/structured-merge-diff/v6/merge)/update.go"

for mutation in dangling owned-version prune owned-items; do
  probe="$formal_root/.lake/source-mutation/$mutation"
  mkdir -p "$probe/SSA"
  cp -R "$formal_root/.lake/build/lib/lean/SSA/." "$probe/SSA/"

  # Each edit preserves supported syntax and changes one source operation.
  python3 - "$updater_source" "$probe/update.go" "$mutation" <<'PY'
import sys
from pathlib import Path

mutations = {
    "dangling": (
        "mergedSet.Difference(prunedSet).Intersection(last)",
        "mergedSet.Difference(prunedSet).Difference(last)",
    ),
    "owned-version": (
        "prunedSet.EnsureNamedFieldsAreMembers(sc, tr).Union(managed.EnsureNamedFieldsAreMembers(sc, tr))",
        "prunedSet.EnsureNamedFieldsAreMembers(sc, tr).Intersection(managed.EnsureNamedFieldsAreMembers(sc, tr))",
    ),
    "prune": (
        "return s.Converter.Convert(pruned, managers[applyingManager].APIVersion())",
        "return s.Converter.Convert(merged, managers[applyingManager].APIVersion())",
    ),
    "owned-items": (
        "delete(managedAtVersion, prunedVersion)",
        'delete(managedAtVersion, "")',
    ),
}
source = Path(sys.argv[1]).read_text()
original, replacement = mutations[sys.argv[3]]
if source.count(original) != 1:
    raise SystemExit(f"Cannot locate unique {sys.argv[3]} source mutation; update verify-source.sh.")
Path(sys.argv[2]).write_text(source.replace(original, replacement))
PY
  go run ./cmd/ssa-source -source "$probe/update.go" >"$probe/SSA/GoSource.lean"
  for module in SourceContracts PruneSource OwnedSource PruningRefinement; do
    cp "SSA/$module.lean" "$probe/SSA/$module.lean"
  done
  for module in GoSource SourceContracts PruneSource OwnedSource PruningRefinement; do
    rm -f "$probe/SSA/$module.olean" "$probe/SSA/$module.olean.private" "$probe/SSA/$module.olean.server"
  done
  case "$mutation" in
    dangling|owned-version) expected=SourceContracts ;;
    prune) expected=PruneSource ;;
    owned-items) expected=OwnedSource ;;
  esac
  (
    cd "$probe"
    export LEAN_PATH="$probe:$formal_root/.lake/build/lib/lean"
    "$lean_binary" -DwarningAsError=true -o SSA/GoSource.olean SSA/GoSource.lean
    rejected=0
    for module in SourceContracts PruneSource OwnedSource; do
      if "$lean_binary" -DwarningAsError=true -o "SSA/$module.olean" "SSA/$module.lean" >"$module.log" 2>&1; then
        continue
      fi
      if [[ "$module" != "$expected" ]] || ! grep -Eq 'unsolved goals|Tactic.*failed' "$module.log"; then
        cat "$module.log" >&2
        printf 'Source mutation %s failed for an unrelated reason in %s.\n' "$mutation" "$module" >&2
        exit 1
      fi
      rejected=1
      break
    done
    if [[ "$rejected" != 1 ]]; then
      printf 'Source mutation %s unexpectedly satisfied the unchanged theorems.\n' "$mutation" >&2
      exit 1
    fi
  )
  printf 'The unchanged Lean theorems correctly rejected the %s Go source mutation.\n' "$mutation"
done
