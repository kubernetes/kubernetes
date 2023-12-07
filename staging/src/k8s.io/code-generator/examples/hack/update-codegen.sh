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

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
SCRIPT_ROOT="${SCRIPT_DIR}/.."
CODEGEN_PKG="${CODEGEN_PKG:-"${SCRIPT_ROOT}/.."}"

source "${CODEGEN_PKG}/kube_codegen.sh"

# generate the code with:
# - --output-base because this script should also be able to run inside the vendor dir of
#   k8s.io/kubernetes. The output-base is needed for the generators to output into the vendor dir
#   instead of the $GOPATH directly. For normal projects this can be dropped.

kube::codegen::gen_helpers \
    --input-pkg-root k8s.io/code-generator/examples \
    --output-base "$(dirname "${BASH_SOURCE[0]}")/../../../.." \
    --boilerplate "${SCRIPT_ROOT}/hack/boilerplate.go.txt"

if [[ -n "${API_KNOWN_VIOLATIONS_DIR:-}" ]]; then
    report_filename="${API_KNOWN_VIOLATIONS_DIR}/codegen_violation_exceptions.list"
    if [[ "${UPDATE_API_KNOWN_VIOLATIONS:-}" == "true" ]]; then
        update_report="--update-report"
    fi
fi

kube::codegen::gen_openapi \
    --input-pkg-root k8s.io/code-generator/examples/apiserver/apis \
    --output-pkg-root k8s.io/code-generator/examples/apiserver \
    --output-base "$(dirname "${BASH_SOURCE[0]}")/../../../.." \
    --report-filename "${report_filename:-"/dev/null"}" \
    ${update_report:+"${update_report}"} \
    --boilerplate "${SCRIPT_ROOT}/hack/boilerplate.go.txt"

kube::codegen::gen_client \
    --with-watch \
    --input-pkg-root k8s.io/code-generator/examples/apiserver/apis \
    --output-pkg-root k8s.io/code-generator/examples/apiserver \
    --output-base "$(dirname "${BASH_SOURCE[0]}")/../../../.." \
    --boilerplate "${SCRIPT_ROOT}/hack/boilerplate.go.txt"

kube::codegen::gen_client \
    --with-watch \
    --with-applyconfig \
    --input-pkg-root k8s.io/code-generator/examples/crd/apis \
    --output-pkg-root k8s.io/code-generator/examples/crd \
    --output-base "$(dirname "${BASH_SOURCE[0]}")/../../../.." \
    --boilerplate "${SCRIPT_ROOT}/hack/boilerplate.go.txt"

kube::codegen::gen_client \
    --with-watch \
    --with-applyconfig \
    --input-pkg-root k8s.io/code-generator/examples/MixedCase/apis \
    --output-pkg-root k8s.io/code-generator/examples/MixedCase \
    --output-base "$(dirname "${BASH_SOURCE[0]}")/../../../.." \
    --boilerplate "${SCRIPT_ROOT}/hack/boilerplate.go.txt"

kube::codegen::gen_client \
    --with-watch \
    --with-applyconfig \
    --input-pkg-root k8s.io/code-generator/examples/HyphenGroup/apis \
    --output-pkg-root k8s.io/code-generator/examples/HyphenGroup \
    --output-base "$(dirname "${BASH_SOURCE[0]}")/../../../.." \
    --boilerplate "${SCRIPT_ROOT}/hack/boilerplate.go.txt"
