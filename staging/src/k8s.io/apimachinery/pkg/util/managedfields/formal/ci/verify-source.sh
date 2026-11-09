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

for mutation in dangling owned-version prune owned-items apply; do
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
    "apply": (
        "fieldpath.NewVersionedSet(set, version, true)",
        "fieldpath.NewVersionedSet(fieldpath.NewSet(), version, true)",
    ),
}
source = Path(sys.argv[1]).read_text()
original, replacement = mutations[sys.argv[3]]
if source.count(original) != 1:
    raise SystemExit(f"Cannot locate unique {sys.argv[3]} source mutation; update verify-source.sh.")
Path(sys.argv[2]).write_text(source.replace(original, replacement))
PY
  go run ./cmd/ssa-source -source "$probe/update.go" >"$probe/SSA/GoSource.lean"
  for module in SourceContracts PruneSource OwnedSource PruningRefinement ApplySource; do
    cp "SSA/$module.lean" "$probe/SSA/$module.lean"
  done
  for module in GoSource SourceContracts PruneSource OwnedSource PruningRefinement ApplySource; do
    rm -f "$probe/SSA/$module.olean" "$probe/SSA/$module.olean.private" "$probe/SSA/$module.olean.server"
  done
  case "$mutation" in
    dangling|owned-version) expected=SourceContracts ;;
    prune) expected=PruneSource ;;
    owned-items) expected=OwnedSource ;;
    apply) expected=ApplySource ;;
  esac
  (
    cd "$probe"
    export LEAN_PATH="$probe:$formal_root/.lake/build/lib/lean"
    "$lean_binary" -DwarningAsError=true -o SSA/GoSource.olean SSA/GoSource.lean
    rejected=0
    for module in SourceContracts PruneSource OwnedSource ApplySource; do
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

# This mutation actually adds a foreign claim: B acquires the actor's newly added field.
probe="$formal_root/.lake/source-mutation/updater-claim-growth"
mkdir -p "$probe/SSA"
cp -R "$formal_root/.lake/build/lib/lean/SSA/." "$probe/SSA/"
python3 - "$updater_source" "$probe/update.go" <<'PY_UPDATER_MUTATION'
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text()
old = "managers[manager].Set().Difference(conflictSet.Set())"
if source.count(old) != 1:
    raise SystemExit("Cannot locate the unique ownership-update mutation; update verify-source.sh.")
Path(sys.argv[2]).write_text(source.replace(old, "managers[manager].Set().Union(conflictSet.Set()).Union(compare.Added)"))
PY_UPDATER_MUTATION
go run ./cmd/ssa-source -updater -source "$probe/update.go" > "$probe/SSA/UpdaterSource.lean"
cp SSA/UpdaterContracts.lean "$probe/SSA/UpdaterContracts.lean"
cp SSA/UpdaterSubset.lean "$probe/SSA/UpdaterSubset.lean"
for module in UpdaterSource UpdaterSubset UpdaterContracts; do
  rm -f "$probe/SSA/$module.olean" "$probe/SSA/$module.olean.private" "$probe/SSA/$module.olean.server"
done
(
  cd "$probe"
  export LEAN_PATH="$probe:$formal_root/.lake/build/lib/lean"
  "$lean_binary" -DwarningAsError=true -o SSA/UpdaterSource.olean SSA/UpdaterSource.lean
  "$lean_binary" -DwarningAsError=true -o SSA/UpdaterSubset.olean SSA/UpdaterSubset.lean
  if "$lean_binary" -DwarningAsError=true -o SSA/UpdaterContracts.olean SSA/UpdaterContracts.lean >UpdaterContracts.log 2>&1; then
    printf 'The ownership-growth mutation unexpectedly satisfied the unchanged source certificate.\n' >&2
    exit 1
  fi
  if ! grep -Eq 'unsolved goals|Tactic.*failed' UpdaterContracts.log; then
    cat UpdaterContracts.log >&2
    printf 'The ownership-growth mutation failed for an unrelated reason.\n' >&2
    exit 1
  fi
)

# The independent native witness must reject the same supported, compilable Go mutation.
python3 - "$updater_source" "$probe" "$formal_root/pr336/updater_test.go.in" <<'PY_UPDATER_OVERLAY'
import json
import sys
from pathlib import Path

source, probe, witness = map(Path, sys.argv[1:])
overlay = {"Replace": {
    str(source): str(probe / "update.go"),
    str(source.parent / "ssa_formal_updater_test.go"): str(witness),
}}
(probe / "overlay.json").write_text(json.dumps(overlay))
PY_UPDATER_OVERLAY
SSA_UPDATER_EXPECTED="$("$formal_root/.lake/build/bin/ssa-model" --updater-observations)"
export SSA_UPDATER_EXPECTED
if go test -overlay="$probe/overlay.json" sigs.k8s.io/structured-merge-diff/v6/merge \
    -run '^TestSSAFormalUpdaterFilteredCache$' -v -count=1 > "$probe/native.log" 2>&1; then
  printf 'Native ownership-growth mutation unexpectedly preserved the foreign claim bound.\n' >&2
  exit 1
fi
if ! grep -q 'forced update created or retained foreign claims' "$probe/native.log"; then
  cat "$probe/native.log" >&2
  printf 'Native ownership-growth mutation failed for an unrelated reason.\n' >&2
  exit 1
fi
printf 'Unchanged Lean certificate and native Go witness both rejected ownership-claim growth.\n'

# Each element method must fail its shared proof when its equal-key guard is flipped.
for method in Difference Union Intersection; do
  probe="$formal_root/.lake/source-mutation/element-$method"
  mkdir -p "$probe/SSA"
  cp -R "$formal_root/.lake/build/lib/lean/SSA/." "$probe/SSA/"
  element_source="$(go list -f '{{.Dir}}' sigs.k8s.io/structured-merge-diff/v6/fieldpath)/element.go"
  python3 - "$element_source" "$probe/element.go" "$method" <<'PY_MUTATION'
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text()
method = sys.argv[3]
start = source.index(f"func (s *PathElementSet) {method}(")
end = source.index("\n}\n", start) + 2
body = source[start:end]
original = "if !s2.members[j].Less(s.members[i]) {"
if body.count(original) != 1:
    raise SystemExit(f"Cannot locate unique element-{method} mutation; update verify-source.sh.")
body = body.replace(original, "if s2.members[j].Less(s.members[i]) {")
Path(sys.argv[2]).write_text(source[:start] + body + source[end:])
PY_MUTATION
  go run ./cmd/ssa-source -elements -source "$probe/element.go" >"$probe/SSA/ElementSource.lean"
  cp SSA/ElementAlgebra.lean "$probe/SSA/ElementAlgebra.lean"
  for module in ElementSource ElementAlgebra; do
    rm -f "$probe/SSA/$module.olean" "$probe/SSA/$module.olean.private" "$probe/SSA/$module.olean.server"
  done
  (
    cd "$probe"
    export LEAN_PATH="$probe:$formal_root/.lake/build/lib/lean"
    "$lean_binary" -DwarningAsError=true -o SSA/ElementSource.olean SSA/ElementSource.lean
    if "$lean_binary" -DwarningAsError=true -o SSA/ElementAlgebra.olean SSA/ElementAlgebra.lean >ElementAlgebra.log 2>&1; then
      printf 'Element-%s mutation unexpectedly satisfied the unchanged theorem.\n' "$method" >&2
      exit 1
    fi
    if ! grep -Eq 'unsolved goals|Tactic.*failed' ElementAlgebra.log; then
      cat ElementAlgebra.log >&2
      printf 'Element-%s mutation failed for an unrelated reason.\n' "$method" >&2
      exit 1
    fi
  )
  printf 'The unchanged Lean theorems correctly rejected the element-%s Go source mutation.\n' "$method"
done
