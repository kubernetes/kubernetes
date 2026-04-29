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

# This script verifies that every declared declarative-validation rule has at
# least one matching observation from a declarative_validation_test.go run.
# Intentional gaps may be suppressed via hack/declarative-validation-coverage-allowlist.yaml.
# Usage: `hack/verify-declarative-validation-coverage.sh [group ...]`.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::golang::setup_env

cd "${KUBE_ROOT}"

# Optional positional args filter both DV-tagged and test packages by group
# name (matched as a directory segment under apis/, api/, or registry/).
# Set OUT_DIR to a writable path to keep artifacts (the declared-rules JSON
# and the per-package observation directory) for inspection; otherwise
# everything lands in a tmpdir cleaned at exit.

if [[ -n "${OUT_DIR:-}" ]]; then
    TMP="${OUT_DIR}"
    kube::log::status "Writing artifacts to OUT_DIR=${TMP} (no cleanup)"
else
    TMP="$(mktemp -d -t verify-dv-coverage.XXXXXX)"
    kube::util::trap_add "rm -rf ${TMP}" EXIT
fi
mkdir -p "${TMP}/actual"

# readonly_pkgs mirrors hack/update-codegen.sh's validation-gen invocation
# (codegen::validation): types used transitively from API types whose
# validations are referenced but not regenerated here.
readonly_pkgs=(
    k8s.io/apimachinery/pkg/apis/meta/v1
    k8s.io/apimachinery/pkg/api/resource
    k8s.io/apimachinery/pkg/runtime
    k8s.io/apimachinery/pkg/types
    k8s.io/apimachinery/pkg/util/intstr
    time
)

# Discover every DV-tagged package (mirrors update-codegen.sh's discovery).
dv_pkgs=()
kube::util::read-array dv_pkgs < <(
    git grep --untracked -l '+k8s:validation-gen=' \
        ':!:vendor/*' ':(glob)pkg/apis/**/doc.go' ':(glob)staging/src/k8s.io/api/**/doc.go' \
        | while read -r f; do echo "./$(dirname "${f}")"; done \
        | sort -u
)
# Discover every package containing a declarative_validation_test.go file.
test_pkgs=()
kube::util::read-array test_pkgs < <(
    git grep --untracked -l . \
        ':(glob)**/declarative_validation_test.go' ':!:vendor/*' \
        | while read -r f; do echo "./$(dirname "${f}")"; done \
        | sort -u
)
# Per-subresource test files don't match the strict glob; list them explicitly.
test_pkgs+=(
    ./pkg/registry/core/replicationcontroller/storage
)

if [[ $# -gt 0 ]]; then
    # matches returns 0 if $1 contains any of the remaining args as a directory
    # segment under apis/, api/, or registry/. Outer loop is over packages so
    # repeated args (e.g. "autoscaling autoscaling") don't duplicate matches.
    matches() {
        local p=$1; shift
        local g
        for g in "$@"; do
            [[ "$p" == */apis/"$g"/* || "$p" == */api/"$g"/* || "$p" == */registry/"$g"/* ]] && return 0
        done
        return 1
    }
    filtered_dv=() filtered_test=()
    for p in "${dv_pkgs[@]}";   do matches "$p" "$@" && filtered_dv+=("$p");   done
    for p in "${test_pkgs[@]}"; do matches "$p" "$@" && filtered_test+=("$p"); done
    if [[ ${#filtered_dv[@]} -eq 0 && ${#filtered_test[@]} -eq 0 ]]; then
        kube::log::error "no DV packages or test packages matched groups: $*"
        exit 1
    fi
    dv_pkgs=("${filtered_dv[@]}")
    test_pkgs=("${filtered_test[@]}")
    kube::log::status "Filtered to groups: $*"
fi

kube::log::status "Found ${#dv_pkgs[@]} DV-tagged packages"
kube::log::status "Found ${#test_pkgs[@]} packages with declarative_validation_test.go"

# 1. Declared rules: one big JSON array of Reports across every (Group, Version).
kube::log::status "Generating declared rules report"
readonly_args=()
for p in "${readonly_pkgs[@]}"; do readonly_args+=(--readonly-pkg "$p"); done
go run ./staging/src/k8s.io/code-generator/cmd/validation-gen \
    --report-rules \
    "${readonly_args[@]}" \
    "${dv_pkgs[@]}" \
    > "${TMP}/expected.json"

# 2. Observed rules: per-package files written by declarative_validation_test.go runs.
# Skip the run when no test packages were found (e.g. group filter matched
# only deprecated/unserved groups); the verifier still runs against an empty
# observation set so allowlists can suppress the resulting "uncovered".
if (( ${#test_pkgs[@]} > 0 )); then
    kube::log::status "Running declarative_validation_test.go suites"
    VALIDATION_RULES_REPORT_DIR="${TMP}/actual" \
        go test -count=1 "${test_pkgs[@]}"
else
    kube::log::status "No declarative_validation_test.go suites to run"
fi

# 3. Diff. The verifier exits non-zero on uncovered rules and prints them.
ALLOWLIST="${KUBE_ROOT}/hack/declarative-validation-coverage-allowlist.yaml"
allowlist_arg=()
if [[ -f "${ALLOWLIST}" ]]; then
    allowlist_arg=(--allowlist="${ALLOWLIST}")
    kube::log::status "Using allowlist: ${ALLOWLIST}"
fi
kube::log::status "Verifying coverage"
go run ./test/declarative_validation verify-coverage \
    --expected-rules="${TMP}/expected.json" \
    --actual-rules-dir="${TMP}/actual" \
    "${allowlist_arg[@]}"
