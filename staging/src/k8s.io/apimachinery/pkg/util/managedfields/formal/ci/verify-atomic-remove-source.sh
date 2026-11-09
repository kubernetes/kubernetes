#!/usr/bin/env bash
# Copyright 2026 The Kubernetes Authors.
# SPDX-License-Identifier: Apache-2.0
set -o errexit
set -o nounset
set -o pipefail
formal_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$formal_root"
SSA_LAKE="${SSA_LAKE:-lake}"
lean_binary="$("$SSA_LAKE" env which lean)"
typed_source="$(go list -f '{{.Dir}}' sigs.k8s.io/structured-merge-diff/v6/typed)/remove.go"
export SSA_SET_MODEL="$formal_root/.lake/build/bin/ssa-model"
for kind in scalar map list factory-extract factory-skip factory-output; do
  probe="$formal_root/.lake/source-mutation/atomic-remove-$kind"
  mkdir -p "$probe/SSA"
  cp -R "$formal_root/.lake/build/lib/lean/SSA/." "$probe/SSA/"
  python3 - "$typed_source" "$probe/remove.go" "$kind" <<'PY_MUTATION'
import sys
from pathlib import Path
source = Path(sys.argv[1]).read_text()
kind = sys.argv[3]
if kind.startswith("factory-"):
    start = source.index("func removeItemsWithSchema(")
else:
    method = {"scalar": "doScalar", "map": "doMap", "list": "doList"}[kind]
    start = source.index(f"func (w *removingWalker) {method}(")
end = source.index("\n}\n", start) + 2
body = source[start:end]
if kind.startswith("factory-"):
    old, new = {
        "factory-extract": ("shouldExtract: shouldExtract", "shouldExtract: !shouldExtract"),
        "factory-skip": ("resolveSchema(schema, typeRef, val, w)", ""),
        "factory-output": ("value.NewValueInterface(w.out)", "value.NewValueInterface(nil)"),
    }[kind]
    if body.count(old) != 1:
        raise SystemExit("Cannot locate unique removal factory mutation")
    body = body.replace(old, new)
elif kind == "scalar":
    old, new = "w.out = w.value.Unstructured()", "w.out = nil"
    if body.count(old) != 1:
        raise SystemExit("Cannot locate scalar removal output mutation")
    body = body.replace(old, new)
else:
    atomic = body.index("if t.ElementRelationship == schema.Atomic")
    head, tail = body[:atomic], body[atomic:]
    if "if w.shouldExtract" not in tail:
        raise SystemExit("Cannot locate atomic extraction guard mutation")
    body = head + tail.replace("if w.shouldExtract", "if !w.shouldExtract", 1)
Path(sys.argv[2]).write_text(source[:start] + body + source[end:])
PY_MUTATION
  go run ./cmd/ssa-source -atomic-remove -source "$probe/remove.go" > "$probe/SSA/AtomicRemoveSource.lean"
  for module in AtomicRemoveSubset AtomicRemoveContracts RemoveFactorySubset RemoveFactoryContracts; do cp "SSA/$module.lean" "$probe/SSA/$module.lean"; done
  for module in AtomicRemoveSource AtomicRemoveSubset AtomicRemoveContracts RemoveFactorySubset RemoveFactoryContracts; do
    rm -f "$probe/SSA/$module.olean" "$probe/SSA/$module.olean.private" "$probe/SSA/$module.olean.server"
  done
  expected=AtomicRemoveContracts
  if [[ "$kind" == factory-* ]]; then expected=RemoveFactoryContracts; fi
  (
    cd "$probe"
    export LEAN_PATH="$probe:$formal_root/.lake/build/lib/lean"
    rejected=0
    for module in AtomicRemoveSource AtomicRemoveSubset AtomicRemoveContracts RemoveFactorySubset RemoveFactoryContracts; do
      if "$lean_binary" -DwarningAsError=true -o "SSA/$module.olean" "SSA/$module.lean" > "$module.log" 2>&1; then continue; fi
      if [[ "$module" != "$expected" ]] || ! grep -q 'unsolved goals' "$module.log"; then
        cat "$module.log" >&2
        printf 'Removal mutation failed in an unrelated module or for an unrelated reason.\n' >&2
        exit 1
      fi
      rejected=1
      break
    done
    if [[ "$rejected" != 1 ]]; then
      printf 'Removal mutation unexpectedly satisfied the unchanged theorems.\n' >&2
      exit 1
    fi
  )
  python3 - "$probe" "$typed_source" "$formal_root/pr336/atomic_remove_test.go.in" <<'PY_OVERLAY'
import json
import sys
from pathlib import Path
probe, source, witness = map(Path, sys.argv[1:])
(probe / "overlay.json").write_text(json.dumps({"Replace": {
    str(source): str(probe / "remove.go"),
    str(source.parent / "ssa_formal_atomic_remove_test.go"): str(witness),
}}))
PY_OVERLAY
  test_name=TestSSAFormalAtomicRemove
  test_kind="$kind"
  oracle='atomic removal fragmented a value or changed its call frame'
  if [[ "$kind" == factory-* ]]; then
    test_name=TestSSAFormalRemoveFactory
    test_kind=scalar
    if [[ "$kind" == factory-extract ]]; then test_kind=map; fi
    oracle='removal factory violated fresh-output or wrapping semantics'
  fi
  if bash "$formal_root/ci/run-native-go.sh" "$probe/overlay.json" sigs.k8s.io/structured-merge-diff/v6/typed \
      -run "^$test_name$/$test_kind$" -v -count=1 > "$probe/native.log" 2>&1; then
    printf 'Native atomic removal mutation unexpectedly preserved whole output.\n' >&2
    exit 1
  fi
  if ! grep -q "$oracle" "$probe/native.log"; then
    cat "$probe/native.log" >&2
    printf 'Native atomic removal mutation failed for an unrelated reason.\n' >&2
    exit 1
  fi
  printf 'Unchanged Lean certificate and native Go oracle both rejected the %s removal mutation.\n' "$kind"
done
