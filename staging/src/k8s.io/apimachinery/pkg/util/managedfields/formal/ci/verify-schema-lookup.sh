#!/usr/bin/env bash
# Copyright 2026 The Kubernetes Authors.
# SPDX-License-Identifier: Apache-2.0
set -o errexit
set -o nounset
set -o pipefail
formal_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$formal_root"
export SSA_SET_MODEL="${1:-$formal_root/.lake/build/bin/ssa-model}"
schema_dir="$(go list -f '{{.Dir}}' sigs.k8s.io/structured-merge-diff/v6/schema)"
overlay="$formal_root/.lake/schema-lookup-overlay.json"
python3 - "$overlay" "$schema_dir" "$formal_root/pr336" <<'PY_OVERLAY'
import json
import sys
from pathlib import Path
overlay, schema, fixtures = map(Path, sys.argv[1:])
overlay.write_text(json.dumps({"Replace": {
    str(schema / "ssa_formal_schema_lookup_test.go"): str(fixtures / "schema_lookup_test.go.in"),
}}))
PY_OVERLAY
bash "$formal_root/ci/run-native-go.sh" "$overlay" sigs.k8s.io/structured-merge-diff/v6/schema -run '^TestSSAFormalSchemaLookup$' -v -count=1
