#!/usr/bin/env bash

# Copyright The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -o errexit
set -o nounset
set -o pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
SSA_LAKE="${SSA_LAKE:-lake}"
expected_version="$(sed -n 's/.*:v//p' lean-toolchain)"
actual_version="$("${SSA_LAKE}" env lean --version)"
if [[ "${actual_version}" != "Lean (version ${expected_version},"* ]]; then
  printf 'Expected Lean %s from lean-toolchain; found %s\n' "${expected_version}" "${actual_version}" >&2
  exit 1
fi

updater_source="$(go list -f '{{.Dir}}' sigs.k8s.io/structured-merge-diff/v6/merge)/update.go"
go run ./cmd/ssa-source \
  -source "$updater_source" \
  -check SSA/GoSource.lean
go run ./cmd/ssa-source -updater -source "$updater_source" -check SSA/UpdaterSource.lean
element_source="$(go list -f '{{.Dir}}' sigs.k8s.io/structured-merge-diff/v6/fieldpath)/element.go"
go run ./cmd/ssa-source -elements -source "$element_source" -check SSA/ElementSource.lean
go run ./cmd/ssa-source -has -source "$element_source" -check SSA/HasSource.lean
search_source="$(go list -f '{{.Dir}}' slices)/sort.go"
go run ./cmd/ssa-source -search -source "$search_source" -check SSA/SearchSource.lean
set_source="$(dirname "$element_source")/set.go"
go run ./cmd/ssa-source -sets -source "$set_source" -check SSA/SetSource.lean

"${SSA_LAKE}" clean
"${SSA_LAKE}" build
"${SSA_LAKE}" env leanchecker --fresh Main
bash ci/verify-audit.sh
bash ci/verify-source.sh

"${SSA_LAKE}" exe ssa-model --self-test
"${SSA_LAKE}" exe ssa-model --check fixtures/scenarios.json

# Fixtures are an exported protocol for future Go adapters, not a second hand-maintained model.
"${SSA_LAKE}" exe ssa-model --fixtures >.lake/scenarios.regenerated.json
diff -u fixtures/scenarios.json .lake/scenarios.regenerated.json

go test ./bridge ./cmd/ssa-compare ./cmd/ssa-source
go run ./cmd/ssa-compare -model .lake/build/bin/ssa-model
bash ci/verify-updater.sh
bash ci/verify-sets.sh
bash ci/verify-has.sh
bash ci/verify-pr336.sh
python3 docs/generate.py --check
