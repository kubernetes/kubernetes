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

# Whole-operand replacement and the per-scope leaf guard need independent native witnesses.
merge_typed_source="$(go list -f '{{.Dir}}' sigs.k8s.io/structured-merge-diff/v6/typed)/merge.go"
export SSA_SET_MODEL="$formal_root/.lake/build/bin/ssa-model"
for mutation in right-output-cleared map-leaf-bypass list-leaf-bypass; do
  probe="$formal_root/.lake/source-mutation/atomic-merge-$mutation"
  mkdir -p "$probe/SSA"
  cp -R "$formal_root/.lake/build/lib/lean/SSA/." "$probe/SSA/"
  python3 - "$merge_typed_source" "$probe/merge.go" "$mutation" <<'PY_MERGE_MUTATION'
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text()
if sys.argv[3] == "right-output-cleared":
    start = source.index("ruleKeepRHS = mergeRule(func(w *mergingWalker)")
    end = source.index("\n\t})", start) + 4
    old, new = "w.out = &v", "w.out = nil; _ = v"
else:
    method = "doMap" if sys.argv[3].startswith("map") else "doList"
    start = source.index(f"func (w *mergingWalker) {method}(")
    end = source.index("\n}\n", start) + 2
    old, new = "w.doLeaf()", "w.rule(w)"
body = source[start:end]
if old not in body or (sys.argv[3] != "right-output-cleared" and body.count(old) != 1):
    raise SystemExit("Cannot locate atomic merge source mutation; update verify-source.sh.")
Path(sys.argv[2]).write_text(source[:start] + body.replace(old, new, 1) + source[end:])
PY_MERGE_MUTATION
  go run ./cmd/ssa-source -atomic-merge -source "$probe/merge.go" > "$probe/SSA/AtomicMergeSource.lean"
  for module in AtomicMergeMachine AtomicMergeSubset AtomicMergeContracts; do cp "SSA/$module.lean" "$probe/SSA/$module.lean"; done
  for module in AtomicMergeSource AtomicMergeMachine AtomicMergeSubset AtomicMergeContracts; do
    rm -f "$probe/SSA/$module.olean" "$probe/SSA/$module.olean.private" "$probe/SSA/$module.olean.server"
  done
  (
    cd "$probe"
    export LEAN_PATH="$probe:$formal_root/.lake/build/lib/lean"
    for module in AtomicMergeSource AtomicMergeMachine AtomicMergeSubset; do
      "$lean_binary" -DwarningAsError=true -o "SSA/$module.olean" "SSA/$module.lean"
    done
    if "$lean_binary" -DwarningAsError=true -o SSA/AtomicMergeContracts.olean SSA/AtomicMergeContracts.lean > AtomicMergeContracts.log 2>&1; then
      printf 'Atomic merge mutation unexpectedly satisfied the unchanged certificate.\n' >&2
      exit 1
    fi
    if ! grep -q 'unsolved goals' AtomicMergeContracts.log; then
      cat AtomicMergeContracts.log >&2
      printf 'Atomic merge mutation failed for an unrelated reason.\n' >&2
      exit 1
    fi
  )
  python3 - "$probe" "$merge_typed_source" "$formal_root/pr336/atomic_merge_test.go.in" <<'PY_MERGE_OVERLAY'
import json
import sys
from pathlib import Path

probe, source, witness = map(Path, sys.argv[1:])
(probe / "overlay.json").write_text(json.dumps({"Replace": {
    str(source): str(probe / "merge.go"),
    str(source.parent / "ssa_formal_atomic_merge_test.go"): str(witness),
}}))
PY_MERGE_OVERLAY
  kind=map
  if [[ "$mutation" == list-leaf-bypass ]]; then kind=list; fi
  if bash "$formal_root/ci/run-native-go.sh" "$probe/overlay.json" sigs.k8s.io/structured-merge-diff/v6/typed \
      -run "^TestSSAFormalAtomicMerge$/$kind$" -v -count=1 > "$probe/native.log" 2>&1; then
    printf 'Native atomic merge mutation unexpectedly preserved whole operands and the leaf guard.\n' >&2
    exit 1
  fi
  if ! grep -q 'atomic merge violated whole-operand choice or call frame' "$probe/native.log"; then
    cat "$probe/native.log" >&2
    printf 'Native atomic merge mutation failed for an unrelated reason.\n' >&2
    exit 1
  fi
  printf 'Unchanged Lean certificate and native Go oracle both rejected the %s atomic-merge mutation.\n' "$mutation"
done

# A comparator collision makes the representative label observable: returning the right label invents a path.
probe="$formal_root/.lake/source-mutation/recursive-set-label"
mkdir -p "$probe/SSA"
cp -R "$formal_root/.lake/build/lib/lean/SSA/." "$probe/SSA/"
set_source="$(go list -f '{{.Dir}}' sigs.k8s.io/structured-merge-diff/v6/fieldpath)/set.go"
python3 - "$set_source" "$probe/set.go" <<'PY_SET_MUTATION'
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text()
start = source.index("func (s *SetNodeMap) Difference(")
end = source.index("\n}\n", start) + 2
body = source[start:end]
old = "out.members = append(out.members, setNode{pathElement: s.members[i].pathElement, set: diff})"
if body.count(old) != 1:
    raise SystemExit("Cannot locate the unique recursive-set representative mutation; update verify-source.sh.")
new = "out.members = append(out.members, setNode{pathElement: s2.Children.members[j].pathElement, set: diff})"
Path(sys.argv[2]).write_text(source[:start] + body.replace(old, new) + source[end:])
PY_SET_MUTATION
go run ./cmd/ssa-source -sets -source "$probe/set.go" > "$probe/SSA/SetSource.lean"
for module in SetSubset SetContracts; do cp "SSA/$module.lean" "$probe/SSA/$module.lean"; done
for module in SetSource SetSubset SetContracts; do
  rm -f "$probe/SSA/$module.olean" "$probe/SSA/$module.olean.private" "$probe/SSA/$module.olean.server"
done
(
  cd "$probe"
  export LEAN_PATH="$probe:$formal_root/.lake/build/lib/lean"
  "$lean_binary" -DwarningAsError=true -o SSA/SetSource.olean SSA/SetSource.lean
  "$lean_binary" -DwarningAsError=true -o SSA/SetSubset.olean SSA/SetSubset.lean
  if "$lean_binary" -DwarningAsError=true -o SSA/SetContracts.olean SSA/SetContracts.lean > SetContracts.log 2>&1; then
    printf 'Recursive-set label mutation unexpectedly satisfied the unchanged certificate.\n' >&2
    exit 1
  fi
  if ! grep -Eq 'unsolved goals|Tactic.*failed' SetContracts.log; then
    cat SetContracts.log >&2
    printf 'Recursive-set mutation failed for an unrelated reason.\n' >&2
    exit 1
  fi
)
python3 - "$set_source" "$probe" "$formal_root/pr336/sets_test.go.in" <<'PY_SET_OVERLAY'
import json
import sys
from pathlib import Path

source, probe, witness = map(Path, sys.argv[1:])
overlay = {"Replace": {
    str(source): str(probe / "set.go"),
    str(source.parent / "ssa_formal_sets_test.go"): str(witness),
}}
(probe / "overlay.json").write_text(json.dumps(overlay))
PY_SET_OVERLAY
export SSA_SET_MODEL="$formal_root/.lake/build/bin/ssa-model"
if bash "$formal_root/ci/run-native-go.sh" "$probe/overlay.json" sigs.k8s.io/structured-merge-diff/v6/fieldpath \
    -run '^TestSSAFormalRecursiveDifference$' -v -count=1 > "$probe/native.log" 2>&1; then
  printf 'Native recursive-set mutation unexpectedly preserved exact path non-growth.\n' >&2
  exit 1
fi
if ! grep -q 'recursive Difference invented an exact left path' "$probe/native.log"; then
  cat "$probe/native.log" >&2
  printf 'Native recursive-set mutation failed for an unrelated reason.\n' >&2
  exit 1
fi
printf 'Unchanged Lean certificate and native Go witness both rejected the recursive-set label mutation.\n'

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
if bash "$formal_root/ci/run-native-go.sh" "$probe/overlay.json" sigs.k8s.io/structured-merge-diff/v6/merge \
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

# Ordinary Difference removes a root member but leaves its descendants; it cannot implement a reset filter.
probe="$formal_root/.lake/source-mutation/reset-filter-scope"
mkdir -p "$probe/SSA"
cp -R "$formal_root/.lake/build/lib/lean/SSA/." "$probe/SSA/"
set_source="$(go list -f '{{.Dir}}' sigs.k8s.io/structured-merge-diff/v6/fieldpath)/set.go"
python3 - "$set_source" "$probe/set.go" <<'PY_FILTER_MUTATION'
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text()
old = "return set.RecursiveDifference(t.excludeSet)"
if source.count(old) != 1:
    raise SystemExit("Cannot locate the unique reset-filter mutation; update verify-source.sh.")
Path(sys.argv[2]).write_text(source.replace(old, "return set.Difference(t.excludeSet)"))
PY_FILTER_MUTATION
go run ./cmd/ssa-source -sets -source "$probe/set.go" > "$probe/SSA/SetSource.lean"
for module in SetSubset SetContracts SetFilterContracts; do cp "SSA/$module.lean" "$probe/SSA/$module.lean"; done
for module in SetSource SetSubset SetContracts SetFilterContracts; do
  rm -f "$probe/SSA/$module.olean" "$probe/SSA/$module.olean.private" "$probe/SSA/$module.olean.server"
done
(
  cd "$probe"
  export LEAN_PATH="$probe:$formal_root/.lake/build/lib/lean"
  for module in SetSource SetSubset SetContracts; do
    "$lean_binary" -DwarningAsError=true -o "SSA/$module.olean" "SSA/$module.lean"
  done
  if "$lean_binary" -DwarningAsError=true -o SSA/SetFilterContracts.olean SSA/SetFilterContracts.lean > SetFilterContracts.log 2>&1; then
    printf 'Reset-filter mutation unexpectedly satisfied the unchanged certificate.\n' >&2
    exit 1
  fi
  if ! grep -Eq 'unsolved goals|Tactic.*failed' SetFilterContracts.log; then
    cat SetFilterContracts.log >&2
    printf 'Reset-filter mutation failed for an unrelated reason.\n' >&2
    exit 1
  fi
)
python3 - "$set_source" "$probe" "$formal_root/pr336/sets_test.go.in" <<'PY_FILTER_OVERLAY'
import json
import sys
from pathlib import Path

source, probe, witness = map(Path, sys.argv[1:])
overlay = {"Replace": {
    str(source): str(probe / "set.go"),
    str(source.parent / "ssa_formal_sets_test.go"): str(witness),
}}
(probe / "overlay.json").write_text(json.dumps(overlay))
PY_FILTER_OVERLAY
export SSA_SET_MODEL="$formal_root/.lake/build/bin/ssa-model"
if bash "$formal_root/ci/run-native-go.sh" "$probe/overlay.json" sigs.k8s.io/structured-merge-diff/v6/fieldpath \
    -run '^TestSSAFormalRootResetFilter$' -v -count=1 > "$probe/native.log" 2>&1; then
  printf 'Native reset-filter mutation unexpectedly excluded complete subtrees.\n' >&2
  exit 1
fi
if ! grep -q 'reset filter retained an excluded prefix' "$probe/native.log"; then
  cat "$probe/native.log" >&2
  printf 'Native reset-filter mutation failed for an unrelated reason.\n' >&2
  exit 1
fi
printf 'Unchanged Lean certificate and native Go witness both rejected the reset-filter scope mutation.\n'

# The standard-library lookup's final equality is part of Has, rather than a trusted membership oracle.
probe="$formal_root/.lake/source-mutation/search-membership"
mkdir -p "$probe/SSA"
cp -R "$formal_root/.lake/build/lib/lean/SSA/." "$probe/SSA/"
search_source="$(go list -f '{{.Dir}}' slices)/sort.go"
python3 - "$search_source" "$probe/sort.go" <<'PY_SEARCH_MUTATION'
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text()
start = source.index("func BinarySearchFunc[")
end = source.index("\n}\n", start) + 2
body = source[start:end]
old = "cmp(x[i], target) == 0"
if body.count(old) != 1:
    raise SystemExit("Cannot locate the unique standard-search mutation; update verify-source.sh.")
Path(sys.argv[2]).write_text(source[:start] + body.replace(old, "cmp(x[i], target) != 0") + source[end:])
PY_SEARCH_MUTATION
go run ./cmd/ssa-source -search -source "$probe/sort.go" > "$probe/SSA/SearchSource.lean"
for module in SearchSubset SearchIndex SearchAlgorithm SearchContracts; do cp "SSA/$module.lean" "$probe/SSA/$module.lean"; done
for module in SearchSource SearchSubset SearchIndex SearchAlgorithm SearchContracts; do
  rm -f "$probe/SSA/$module.olean" "$probe/SSA/$module.olean.private" "$probe/SSA/$module.olean.server"
done
(
  cd "$probe"
  export LEAN_PATH="$probe:$formal_root/.lake/build/lib/lean"
  for module in SearchSource SearchSubset SearchIndex SearchAlgorithm; do
    "$lean_binary" -DwarningAsError=true -o "SSA/$module.olean" "SSA/$module.lean"
  done
  if "$lean_binary" -DwarningAsError=true -o SSA/SearchContracts.olean SSA/SearchContracts.lean > SearchContracts.log 2>&1; then
    printf 'Standard-search mutation unexpectedly satisfied the unchanged certificate.\n' >&2
    exit 1
  fi
  if ! grep -Eq 'unsolved goals|Tactic.*failed' SearchContracts.log; then
    cat SearchContracts.log >&2
    printf 'Standard-search mutation failed for an unrelated reason.\n' >&2
    exit 1
  fi
)
fieldpath_dir="$(go list -f '{{.Dir}}' sigs.k8s.io/structured-merge-diff/v6/fieldpath)"
python3 - "$search_source" "$probe" "$fieldpath_dir" "$formal_root/pr336/has_test.go.in" <<'PY_SEARCH_OVERLAY'
import json
import shutil
import sys
import tempfile
from pathlib import Path

source, probe, fieldpath, witness = map(Path, sys.argv[1:])
# Go forbids overlays within GOMODCACHE, including downloaded toolchain source.
# A shallow GOROOT mirror gives slices real local files and leaves the installed toolchain untouched.
original = source.parents[2]
mirror = Path(tempfile.mkdtemp(prefix="go-root-", dir=probe))
for entry in original.iterdir():
    if entry.name != "src":
        (mirror / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
(mirror / "src").mkdir()
for entry in (original / "src").iterdir():
    if entry.name != "slices":
        (mirror / "src" / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
shutil.copytree(source.parent, mirror / "src/slices")
(probe / "go-root-path").write_text(str(mirror))
(probe / "go-tool-path").write_text(str(original / "bin/go"))
overlay = {"Replace": {
    str(mirror / "src/slices/sort.go"): str(probe / "sort.go"),
    str(fieldpath / "ssa_formal_has_test.go"): str(witness),
}}
(probe / "overlay.json").write_text(json.dumps(overlay))
PY_SEARCH_OVERLAY
export SSA_SET_MODEL="$formal_root/.lake/build/bin/ssa-model"
search_go_root="$(cat "$probe/go-root-path")"
search_go_tool="$(cat "$probe/go-tool-path")"
if GOROOT="$search_go_root" GOTOOLCHAIN=local SSA_NATIVE_GO="$search_go_tool" \
    bash "$formal_root/ci/run-native-go.sh" "$probe/overlay.json" sigs.k8s.io/structured-merge-diff/v6/fieldpath \
    -run '^TestSSAFormalHasSearch$' -v -count=1 > "$probe/native.log" 2>&1; then
  printf 'Native standard-search mutation unexpectedly preserved exact membership.\n' >&2
  exit 1
fi
if ! grep -q 'Has violated exact membership' "$probe/native.log"; then
  cat "$probe/native.log" >&2
  printf 'Native standard-search mutation failed for an unrelated reason.\n' >&2
  exit 1
fi
printf 'Unchanged Lean certificate and native Go witness both rejected the standard-library membership mutation.\n'

# Each atomic guard independently protects whole-scope ownership, even for null values.
export SSA_SET_MODEL="$formal_root/.lake/build/bin/ssa-model"
typed_source="$(go list -f '{{.Dir}}' sigs.k8s.io/structured-merge-diff/v6/typed)/tofieldset.go"
for method in doMap doList; do
  probe="$formal_root/.lake/source-mutation/atomic-fields-$method"
  mkdir -p "$probe/SSA"
  cp -R "$formal_root/.lake/build/lib/lean/SSA/." "$probe/SSA/"
  python3 - "$typed_source" "$probe/tofieldset.go" "$method" <<'PY_ATOMIC_MUTATION'
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text()
start = source.index(f"func (v *toFieldSetWalker) {sys.argv[3]}(")
end = source.index("\n}\n", start) + 2
body = source[start:end]
original = "t.ElementRelationship == schema.Atomic"
if body.count(original) != 1:
    raise SystemExit("Cannot locate unique typed atomic guard; update verify-source.sh.")
Path(sys.argv[2]).write_text(source[:start] + body.replace(original, "t.ElementRelationship != schema.Atomic") + source[end:])
PY_ATOMIC_MUTATION
  go run ./cmd/ssa-source -atomic-fields -source "$probe/tofieldset.go" > "$probe/SSA/AtomicFieldSource.lean"
  for module in AtomicFieldSubset AtomicFieldContracts; do cp "SSA/$module.lean" "$probe/SSA/$module.lean"; done
  for module in AtomicFieldSource AtomicFieldSubset AtomicFieldContracts; do
    rm -f "$probe/SSA/$module.olean" "$probe/SSA/$module.olean.private" "$probe/SSA/$module.olean.server"
  done
  (
    cd "$probe"
    export LEAN_PATH="$probe:$formal_root/.lake/build/lib/lean"
    for module in AtomicFieldSource AtomicFieldSubset; do
      "$lean_binary" -DwarningAsError=true -o "SSA/$module.olean" "SSA/$module.lean"
    done
    if "$lean_binary" -DwarningAsError=true -o SSA/AtomicFieldContracts.olean SSA/AtomicFieldContracts.lean > AtomicFieldContracts.log 2>&1; then
      printf 'Typed atomic mutation unexpectedly satisfied its unchanged certificate.\n' >&2
      exit 1
    fi
    if ! grep -q 'unsolved goals' AtomicFieldContracts.log; then
      cat AtomicFieldContracts.log >&2
      printf 'Typed atomic mutation failed for an unrelated reason.\n' >&2
      exit 1
    fi
  )
  python3 - "$probe" "$typed_source" "$formal_root/pr336/atomic_fields_test.go.in" <<'PY_ATOMIC_OVERLAY'
import json
import sys
from pathlib import Path

probe, source, witness = map(Path, sys.argv[1:])
(probe / "overlay.json").write_text(json.dumps({"Replace": {
    str(source): str(probe / "tofieldset.go"),
    str(source.parent / "ssa_formal_atomic_fields_test.go"): str(witness),
}}))
PY_ATOMIC_OVERLAY
  kind=map
  if [[ "$method" == doList ]]; then kind=list; fi
  if bash "$formal_root/ci/run-native-go.sh" "$probe/overlay.json" sigs.k8s.io/structured-merge-diff/v6/typed \
      -run "^TestSSAFormalAtomicFields$/$kind$" -v -count=1 > "$probe/native.log" 2>&1; then
    printf 'Native typed atomic mutation unexpectedly preserved scope claims.\n' >&2
    exit 1
  fi
  if ! grep -q 'atomic extraction fieldset mismatch' "$probe/native.log"; then
    cat "$probe/native.log" >&2
    printf 'Native typed atomic mutation failed for an unrelated reason.\n' >&2
    exit 1
  fi
  printf 'Unchanged Lean certificate and native Go oracle both rejected the %s atomic-guard mutation.\n' "$method"
done
