#!/usr/bin/env bash

# Copyright 2017 The Kubernetes Authors.
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

SCRIPT_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
CODEGEN_PKG=${CODEGEN_PKG:-$(cd "${SCRIPT_ROOT}"; ls -d -1 ./vendor/k8s.io/code-generator 2>/dev/null || echo ../code-generator)}

source "${CODEGEN_PKG}/kube_codegen.sh"

kube::codegen::gen_helpers \
    --input-pkg-root k8s.io/apiextensions-apiserver/pkg \
    --output-base "$(dirname "${BASH_SOURCE[0]}")/../../.." \
    --boilerplate "${SCRIPT_ROOT}/hack/boilerplate.go.txt"

if [[ -n "${API_KNOWN_VIOLATIONS_DIR:-}" ]]; then
    report_filename="${API_KNOWN_VIOLATIONS_DIR}/apiextensions_violation_exceptions.list"
    if [[ "${UPDATE_API_KNOWN_VIOLATIONS:-}" == "true" ]]; then
        update_report="--update-report"
    fi
fi

kube::codegen::gen_openapi \
    --input-pkg-root k8s.io/apiextensions-apiserver/pkg \
    --extra-pkgs k8s.io/api/autoscaling/v1 `# needed for Scale type` \
    --output-pkg-root k8s.io/apiextensions-apiserver/pkg/generated \
    --output-base "$(dirname "${BASH_SOURCE[0]}")/../../.." \
    --report-filename "${report_filename:-"/dev/null"}" \
    ${update_report:+"${update_report}"} \
    --boilerplate "${SCRIPT_ROOT}/hack/boilerplate.go.txt"

kube::codegen::gen_client \
    --with-watch \
    --with-applyconfig \
    --input-pkg-root k8s.io/apiextensions-apiserver/pkg/apis \
    --output-pkg-root k8s.io/apiextensions-apiserver/pkg/client \
    --output-base "$(dirname "${BASH_SOURCE[0]}")/../../.." \
    --versioned-name clientset \
    --boilerplate "${SCRIPT_ROOT}/hack/boilerplate.go.txt"
