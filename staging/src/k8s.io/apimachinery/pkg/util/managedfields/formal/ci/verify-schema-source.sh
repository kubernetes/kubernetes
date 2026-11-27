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
typed_source="$(go list -f '{{.Dir}}' sigs.k8s.io/structured-merge-diff/v6/typed)/helpers.go"
export SSA_SET_MODEL="$formal_root/.lake/build/bin/ssa-model"
for mutation in deduce-scalar-whole handler-priority resolver-skip; do
  probe="$formal_root/.lake/source-mutation/schema-$mutation"
  mkdir -p "$probe/SSA"
  cp -R "$formal_root/.lake/build/lib/lean/SSA/." "$probe/SSA/"
  python3 - "$typed_source" "$probe/helpers.go" "$mutation" <<'PY_MUTATION'
import sys
from pathlib import Path
source = Path(sys.argv[1]).read_text()
old, new = {
    "deduce-scalar-whole": ("return schema.Atom{Scalar: atom.Scalar}", "return atom"),
    "handler-priority": ("case a.Map != nil:\n\t\treturn ah.doMap(a.Map)\n\tcase a.Scalar != nil:\n\t\treturn ah.doScalar(a.Scalar)",
        "case a.Scalar != nil:\n\t\treturn ah.doScalar(a.Scalar)\n\tcase a.Map != nil:\n\t\treturn ah.doMap(a.Map)"),
    "resolver-skip": ("\ta = deduceAtom(a, v)\n", ""),
}[sys.argv[3]]
if source.count(old) != 1:
    raise SystemExit("Cannot locate unique schema source mutation")
Path(sys.argv[2]).write_text(source.replace(old, new))
PY_MUTATION
  go run ./cmd/ssa-source -schema-dispatch -source "$probe/helpers.go" > "$probe/SSA/SchemaDispatchSource.lean"
  for module in SchemaDispatchMachine SchemaDispatchSubset SchemaDispatchContracts SchemaDispatchHandler SchemaDispatchResolver; do
    cp "SSA/$module.lean" "$probe/SSA/$module.lean"
  done
  for module in SchemaDispatchSource SchemaDispatchMachine SchemaDispatchSubset SchemaDispatchContracts SchemaDispatchHandler SchemaDispatchResolver; do
    rm -f "$probe/SSA/$module.olean" "$probe/SSA/$module.olean.private" "$probe/SSA/$module.olean.server"
  done
  case "$mutation" in
    deduce-scalar-whole) expected=SchemaDispatchContracts; operation=deduce; oracle='source deduction violated pointer/category semantics' ;;
    handler-priority) expected=SchemaDispatchHandler; operation=handle; oracle='source schema dispatch changed handler, pointer, errors or state' ;;
    resolver-skip) expected=SchemaDispatchResolver; operation=resolve; oracle='source schema dispatch changed handler, pointer, errors or state' ;;
  esac
  (
    cd "$probe"
    export LEAN_PATH="$probe:$formal_root/.lake/build/lib/lean"
    rejected=0
    for module in SchemaDispatchSource SchemaDispatchMachine SchemaDispatchSubset SchemaDispatchContracts SchemaDispatchHandler SchemaDispatchResolver; do
      if "$lean_binary" -DwarningAsError=true -o "SSA/$module.olean" "SSA/$module.lean" > "$module.log" 2>&1; then continue; fi
      if [[ "$module" != "$expected" ]] || ! grep -q 'unsolved goals' "$module.log"; then
        cat "$module.log" >&2
        printf 'Schema mutation failed in an unrelated module or for an unrelated reason.\n' >&2
        exit 1
      fi
      rejected=1
      break
    done
    if [[ "$rejected" != 1 ]]; then
      printf 'Schema mutation unexpectedly satisfied its unchanged source theorem.\n' >&2
      exit 1
    fi
  )
  python3 - "$probe" "$typed_source" "$formal_root/pr336" <<'PY_OVERLAY'
import json
import sys
from pathlib import Path
probe, source, fixtures = map(Path, sys.argv[1:])
(probe / "overlay.json").write_text(json.dumps({"Replace": {
    str(source): str(probe / "helpers.go"),
    str(source.parent / "ssa_formal_schema_dispatch_test.go"): str(fixtures / "schema_dispatch_test.go.in"),
    str(source.parent / "ssa_formal_atomic_remove_test.go"): str(fixtures / "atomic_remove_test.go.in"),
}}))
PY_OVERLAY
  if bash "$formal_root/ci/run-native-go.sh" "$probe/overlay.json" sigs.k8s.io/structured-merge-diff/v6/typed \
      -run "^TestSSAFormalSchemaDispatch$/$operation$" -v -count=1 > "$probe/native.log" 2>&1; then
    printf 'Native schema mutation unexpectedly preserved type selection and errors.\n' >&2
    exit 1
  fi
  if ! grep -q "$oracle" "$probe/native.log"; then
    cat "$probe/native.log" >&2
    printf 'Native schema mutation failed for an unrelated reason.\n' >&2
    exit 1
  fi
  printf 'Unchanged Lean certificate and independent native oracle both rejected schema mutation %s.\n' "$mutation"
done
