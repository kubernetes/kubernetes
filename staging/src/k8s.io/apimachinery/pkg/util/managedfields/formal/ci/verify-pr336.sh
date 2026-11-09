#!/usr/bin/env bash
# Copyright 2026 The Kubernetes Authors.
# SPDX-License-Identifier: Apache-2.0
set -o errexit
set -o nounset
set -o pipefail
formal_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
yaml_root="$(cd "$formal_root" && go list -f '{{.Dir}}' sigs.k8s.io/yaml)"
model="$formal_root/.lake/build/bin/ssa-model"
SSA_UPDATER_EXPECTED="$("$model" --updater-observations)"
export SSA_UPDATER_EXPECTED
export SSA_SET_MODEL="$model"
# These immutable commits isolate this PR from later upstream behavior.
for revision in head base; do
  if [[ "$revision" == head ]]; then
    sha=6fab440a518a1b329e5cabd5a3c924d6c267828b
    directory="$formal_root/.lake/pr336-source"
    export SSA_PR336_BASE=0
  else
    sha=560aaa513ea735bf59e6011feecd5388e2dfeae9
    directory="$formal_root/.lake/pr336-base"
    export SSA_PR336_BASE=1
  fi
  if [[ ! -d "$directory/.git" ]]; then
    git clone --no-checkout https://github.com/kubernetes-sigs/structured-merge-diff.git "$directory"
  fi
  if ! git -C "$directory" cat-file -e "$sha^{commit}"; then
    git -C "$directory" fetch --depth=1 origin "$sha"
  fi
  git -C "$directory" diff --exit-code
  git -C "$directory" checkout --detach "$sha"
  [[ "$(git -C "$directory" rev-parse HEAD)" == "$sha" ]]
  mkdir -p "$directory/formalbridge/yaml" "$directory/formalcmd"
  for file in protocol.go schema.go compare.go numeric_codec.go granular_keys.go; do
    sed -e 's,structured-merge-diff/v6,structured-merge-diff/v7,g' \
      -e 's,"sigs.k8s.io/yaml","sigs.k8s.io/structured-merge-diff/v7/formalbridge/yaml",g' \
      "$formal_root/bridge/$file" > "$directory/formalbridge/$file"
  done
  # Reuse Kubernetes' pinned YAML decoder without changing either module graph.
  cp "$yaml_root/yaml.go" "$yaml_root/fields.go" "$directory/formalbridge/yaml/"
  cp "$formal_root/bridge/schema.yaml" "$formal_root/bridge/structural.yaml" "$directory/formalbridge/"
  cp "$formal_root/pr336/engine.go.in" "$directory/formalbridge/go.go"
  cp "$formal_root/pr336/main.go.in" "$directory/formalcmd/main.go"
  sed 's,structured-merge-diff/v6,structured-merge-diff/v7,g' \
    "$formal_root/pr336/updater_test.go.in" > "$directory/formalbridge/updater_test.go.in"
  sed 's,structured-merge-diff/v6,structured-merge-diff/v7,g' \
    "$formal_root/pr336/sets_test.go.in" > "$directory/formalbridge/sets_test.go.in"
  sed 's,structured-merge-diff/v6,structured-merge-diff/v7,g' \
    "$formal_root/pr336/has_test.go.in" > "$directory/formalbridge/has_test.go.in"
  sed 's,structured-merge-diff/v6,structured-merge-diff/v7,g' \
    "$formal_root/pr336/atomic_fields_test.go.in" > "$directory/formalbridge/atomic_fields_test.go.in"
  sed 's,structured-merge-diff/v6,structured-merge-diff/v7,g' \
    "$formal_root/pr336/atomic_merge_test.go.in" > "$directory/formalbridge/atomic_merge_test.go.in"
  python3 - "$directory" <<'PY_UPDATER'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
overlay = {"Replace": {str(root / "merge/ssa_formal_updater_test.go"): str(root / "formalbridge/updater_test.go.in"),
    str(root / "fieldpath/ssa_formal_sets_test.go"): str(root / "formalbridge/sets_test.go.in"),
    str(root / "fieldpath/ssa_formal_has_test.go"): str(root / "formalbridge/has_test.go.in"),
    str(root / "typed/ssa_formal_atomic_fields_test.go"): str(root / "formalbridge/atomic_fields_test.go.in"),
    str(root / "typed/ssa_formal_atomic_merge_test.go"): str(root / "formalbridge/atomic_merge_test.go.in"),
}}
(root / "formalbridge/updater-overlay.json").write_text(json.dumps(overlay))
PY_UPDATER
  (
    cd "$directory"
    export GOWORK=off
    # A caller can share an ordinary Go cache; sandboxed runs can set task-local paths.
    go test -overlay="$directory/formalbridge/updater-overlay.json" ./merge ./fieldpath ./typed
    go run ./formalcmd "$model" "$formal_root/testdata/pr336.yaml"
  )
done
