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

# This script replace "net" stdlib IP and CIDR parsers
# with the ones forked in k8s.io/utils/net to parse IP addresses
# because of the compatibility break introduced in golang 1.17
# Reference: #100895
# Usage: `hack/update-netparse-cve.sh`.

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
echo 'installing net parser converter'
pushd "${KUBE_ROOT}/hack/tools" >/dev/null
  go install github.com/aojea/sloppy-netparser
popd >/dev/null

cd "${KUBE_ROOT}"

find_files() {
  find . -not \( \
      \( \
        -wholename './output' \
        -o -wholename './.git' \
        -o -wholename './_output' \
        -o -wholename './_gopath' \
        -o -wholename './release' \
        -o -wholename './target' \
        -o -wholename '*/third_party/*' \
        -o -wholename '*/vendor/*' \
        -o -wholename './staging/src/k8s.io/client-go/*vendor/*' \
      \) -prune \
    \) -name '*.go'
}

# replace net.ParseIP() and netParseIPCDR
find_files | xargs sloppy-netparser

