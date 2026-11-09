#!/usr/bin/env bash
# Copyright 2026 The Kubernetes Authors.
# SPDX-License-Identifier: Apache-2.0
set -o errexit
set -o nounset
set -o pipefail

formal_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$formal_root"
export SSA_SET_MODEL="${1:-$formal_root/.lake/build/bin/ssa-model}"
typed_dir="$(go list -f '{{.Dir}}' sigs.k8s.io/structured-merge-diff/v6/typed)"
overlay="$formal_root/.lake/atomic-fields-overlay.json"
python3 - "$overlay" "$typed_dir/ssa_formal_atomic_fields_test.go" "$formal_root/pr336/atomic_fields_test.go.in" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({"Replace": {sys.argv[2]: sys.argv[3]}}))
PY
go test -overlay="$overlay" sigs.k8s.io/structured-merge-diff/v6/typed -run '^TestSSAFormalAtomicFields$' -v -count=1
