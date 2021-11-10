#!/usr/bin/env bash

# Copyright 2021 The Kubernetes Authors.
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

# This script checks coding style for go language files in each
# Kubernetes package by golint.
# Usage: `hack/verify-golangci-lint.sh`.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"
source "${KUBE_ROOT}/hack/lib/util.sh"

kube::golang::verify_go_version

# Ensure that we find the binaries we build before anything else.
export GOBIN="${KUBE_OUTPUT_BINPATH}"
PATH="${GOBIN}:${PATH}"

# Explicitly opt into go modules, even though we're inside a GOPATH directory
export GO111MODULE=on

# Install golangci-lint
echo 'installing golangci-lint '
pushd "${KUBE_ROOT}/hack/tools" >/dev/null
  go install github.com/golangci/golangci-lint/cmd/golangci-lint
popd >/dev/null

cd "${KUBE_ROOT}"

# The config is in ${KUBE_ROOT}/.golangci.yaml
RET=0
if [[ "$#" -gt 0 ]]; then
    echo "running golangci-lint $@"
    if ! golangci-lint run "$@"; then
        RET=1
    fi
else
    echo "running golangci-lint for module $(go list -m)"
    if ! golangci-lint run ./... ; then
        RET=1
    fi
    for d in staging/src/k8s.io/*; do
        pushd "./vendor/k8s.io/$(basename "$d")" >/dev/null
        echo "running golangci-lint for module $(go list -m)"
        if ! golangci-lint run ./... ; then
            RET=1
        fi
        popd >/dev/null
    done
fi
exit $RET
