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
schema_source="$(go list -f '{{.Dir}}' sigs.k8s.io/structured-merge-diff/v6/schema)/elements.go"
export SSA_SET_MODEL="$formal_root/.lake/build/bin/ssa-model"
modules=(SchemaLookupSource SchemaLookupSubset SchemaLookupContracts SchemaResolveMachine SchemaResolveSubset SchemaResolveContracts MapCopySubset MapCopyContracts)
if [[ $# == 0 ]]; then
  set -- missing-name map-relationship cache-hit deferred-unlock alias-after-publish copy-fields copy-once-reset
fi
for mutation in "$@"; do
  probe="$formal_root/.lake/source-mutation/lookup-$mutation"
  mkdir -p "$probe/SSA"
  cp -R "$formal_root/.lake/build/lib/lean/SSA/." "$probe/SSA/"
  python3 - "$schema_source" "$probe/elements.go" "$mutation" <<'PY_MUTATION'
import sys
from pathlib import Path
source = Path(sys.argv[1]).read_text()
old, new = {
    "missing-name": ("if !ok {\n\t\t\treturn Atom{}, false", "if !ok {\n\t\t\treturn tr.Inlined, true"),
    "map-relationship": ("\t\t\t\tmapCopy.ElementRelationship = *tr.ElementRelationship\n", ""),
    "cache-hit": ("if result, exists = s.resolvedTypes[tr]; !exists {", "if result, exists = s.resolvedTypes[tr]; exists {"),
    "deferred-unlock": ("\tdefer s.lock.Unlock()\n", ""),
    "alias-after-publish": ("\t\t\t\tresult.Map = &mapCopy\n", "\t\t\t\tresult.Map = &mapCopy\n\t\t\t\tmapCopy = Map{}\n"),
    "copy-fields": ("\tdst.Fields = m.Fields\n", ""),
    "copy-once-reset": ("\t\tdst.once = sync.Once{}\n\t\tdst.once.Do(func() {\n\t\t\tdst.m.Store(mm)",
        "\t\tdst.once.Do(func() {\n\t\t\tdst.m.Store(mm)"),
}[sys.argv[3]]
if source.count(old) != 1:
    raise SystemExit("Cannot locate unique schema lookup mutation")
Path(sys.argv[2]).write_text(source.replace(old, new))
PY_MUTATION
  go run ./cmd/ssa-source -schema-lookup -source "$probe/elements.go" > "$probe/SSA/SchemaLookupSource.lean"
  for module in "${modules[@]:1}"; do cp "SSA/$module.lean" "$probe/SSA/$module.lean"; done
  for module in "${modules[@]}"; do
    rm -f "$probe/SSA/$module.olean" "$probe/SSA/$module.olean.private" "$probe/SSA/$module.olean.server"
  done
  expected=SchemaResolveContracts
  native_test='^TestSSAFormalSchemaLookup$'
  oracle='source schema lookup violated result, contents, relationship or input immutability'
  if [[ "$mutation" == missing-name ]]; then expected=SchemaLookupContracts; fi
  if [[ "$mutation" == deferred-unlock ]]; then oracle='schema resolver returned with its mutex held'; fi
  if [[ "$mutation" == alias-after-publish ]]; then oracle='schema map copy lost its field index'; fi
  if [[ "$mutation" == copy-fields ]]; then
    expected=MapCopyContracts
    native_test='^TestSSAFormalMapCopy$'
    oracle='source map copy violated fields, cache or once contract'
  fi
  if [[ "$mutation" == copy-once-reset ]]; then
    expected=MapCopyContracts
    native_test='^TestSSAFormalMapCopy$'
    oracle='source map copy changed shared contents or cache pointer identity'
  fi
  (
    cd "$probe"
    export LEAN_PATH="$probe:$formal_root/.lake/build/lib/lean"
    rejected=0
    for module in "${modules[@]}"; do
      if "$lean_binary" -DwarningAsError=true -o "SSA/$module.olean" "SSA/$module.lean" > "$module.log" 2>&1; then continue; fi
      if [[ "$module" != "$expected" ]] || ! grep -q 'unsolved goals' "$module.log"; then
        cat "$module.log" >&2
        printf 'Schema lookup mutation failed in an unrelated module or for an unrelated reason.\n' >&2
        exit 1
      fi
      rejected=1
      break
    done
    if [[ "$rejected" != 1 ]]; then
      printf 'Schema lookup mutation unexpectedly satisfied its unchanged source theorem.\n' >&2
      exit 1
    fi
  )
  python3 - "$probe" "$schema_source" "$formal_root/pr336/schema_lookup_test.go.in" <<'PY_OVERLAY'
import json
import sys
from pathlib import Path
probe, source, fixture = map(Path, sys.argv[1:])
(probe / "overlay.json").write_text(json.dumps({"Replace": {
    str(source): str(probe / "elements.go"),
    str(source.parent / "ssa_formal_schema_lookup_test.go"): str(fixture),
    str(source.parent / "ssa_formal_map_copy_test.go"): str(fixture.parent / "map_copy_test.go.in"),
}}))
PY_OVERLAY
  if bash "$formal_root/ci/run-native-go.sh" "$probe/overlay.json" sigs.k8s.io/structured-merge-diff/v6/schema \
      -run "$native_test" -v -count=1 > "$probe/native.log" 2>&1; then
    printf 'Native schema lookup mutation unexpectedly preserved its independent oracle.\n' >&2
    exit 1
  fi
  if ! grep -q "$oracle" "$probe/native.log"; then
    cat "$probe/native.log" >&2
    printf 'Native schema lookup mutation failed for an unrelated reason.\n' >&2
    exit 1
  fi
  printf 'Unchanged Lean certificate and independent native oracle both rejected schema lookup mutation %s.\n' "$mutation"
done
