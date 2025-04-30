#!/usr/bin/env bash

# Copyright 2025 The Kubernetes Authors.
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

# This script checks the coding style for the Go language files using
# golangci-lint. Which checks are enabled depends on command line flags. The
# default is a minimal set of checks that all existing code passes without
# issues.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"
source "${KUBE_ROOT}/hack/lib/util.sh"

kube::golang::setup_env
export GOBIN="${KUBE_OUTPUT_BIN}"

golangci_config="${KUBE_ROOT}/hack/kube-api-linter.yaml"

if [ "${golangci_config}" ]; then
  # The relative path to _output/local/bin only works if that actually is the
  # GOBIN. If not, then we have to make a temporary copy of the config and
  # replace the path with an absolute one. This could be done also
  # unconditionally, but the invocation that is printed below is nicer if we
  # don't to do it when not required.
  if grep -q 'path: ../_output/local/bin/' "${golangci_config}" &&
     [ "${GOBIN}" != "${KUBE_ROOT}/_output/local/bin" ]; then
    kube::util::ensure-temp-dir
    patched_golangci_config="${KUBE_TEMP}/$(basename "${golangci_config}")"
    sed -e "s;path: ../_output/local/bin/;path: ${GOBIN}/;" "${golangci_config}" >"${patched_golangci_config}"
    golangci_config="${patched_golangci_config}"
  fi
fi

cd "${KUBE_ROOT}/hack"

echo "installing kube-api-linter into ${GOBIN}"
GOTOOLCHAIN="$(kube::golang::hack_tools_gotoolchain)" go -C "${KUBE_ROOT}/hack/tools/golangci-lint" build -o "${GOBIN}/kube-api-linter.so" -buildmode=plugin sigs.k8s.io/kube-api-linter/pkg/plugin

"${KUBE_ROOT}/hack/verify-golangci-lint.sh" -c ${golangci_config} "$@" ./staging/src/k8s.io/api/...
