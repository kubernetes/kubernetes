#!/usr/bin/env bash
# Copyright 2026 The Kubernetes Authors.
# SPDX-License-Identifier: Apache-2.0
set -o errexit
set -o nounset
set -o pipefail

formal_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$formal_root"
model="${1:-$formal_root/.lake/build/bin/ssa-model}"
SSA_UPDATER_EXPECTED="$("$model" --updater-observations)"
export SSA_UPDATER_EXPECTED
ssa_merge_dir="$(go list -f '{{.Dir}}' sigs.k8s.io/structured-merge-diff/v6/merge)"
overlay="$formal_root/.lake/updater-overlay.json"

# Add the internal-package witness virtually, so the selected dependency stays untouched.
python3 - "$overlay" "$ssa_merge_dir/ssa_formal_updater_test.go" "$formal_root/pr336/updater_test.go.in" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({"Replace": {sys.argv[2]: sys.argv[3]}}))
PY
go test -overlay="$overlay" sigs.k8s.io/structured-merge-diff/v6/merge \
  -run '^TestSSAFormalUpdaterFilteredCache$' -v -count=1
