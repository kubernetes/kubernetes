#!/usr/bin/env bash

# Copyright 2018 The Kubernetes Authors.
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

# This script does a fast type check of kubernetes code for all platforms.
# Usage: `hack/verify-typecheck.sh`.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::golang::verify_go_version

cd "${KUBE_ROOT}"

ret=0
TYPECHECK_SERIAL="${TYPECHECK_SERIAL:-false}"

SERVER_PLATFORMS=$(echo "${KUBE_SUPPORTED_SERVER_PLATFORMS[@]}" | tr ' ' ',')
CLIENT_PLATFORMS=$(echo "${KUBE_SUPPORTED_CLIENT_PLATFORMS[@]}" | tr ' ' ',')
NODE_PLATFORMS=$(echo "${KUBE_SUPPORTED_NODE_PLATFORMS[@]}" | tr ' ' ',')
TEST_PLATFORMS=$(echo "${KUBE_SUPPORTED_TEST_PLATFORMS[@]}" | tr ' ' ',')

# As of June, 2020 the typecheck tool is written in terms of go/packages, but
# that library doesn't work well with multiple modules.  Until that is done,
# force this tooling to run in a fake GOPATH.
hack/run-in-gopath.sh \
    go run test/typecheck/main.go "$@" --serial="${TYPECHECK_SERIAL}" --platform "${SERVER_PLATFORMS}" "${KUBE_SERVER_TARGETS[@]}" \
    || ret=$?
hack/run-in-gopath.sh \
    go run test/typecheck/main.go "$@" --serial="${TYPECHECK_SERIAL}" --platform "${CLIENT_PLATFORMS}" "${KUBE_CLIENT_TARGETS[@]}" \
    || ret=$?
hack/run-in-gopath.sh \
    go run test/typecheck/main.go "$@" --serial="${TYPECHECK_SERIAL}" --platform "${NODE_PLATFORMS}" "${KUBE_NODE_TARGETS[@]}" \
    || ret=$?

# $KUBE_TEST_TARGETS doesn't seem to work like the other TARGETS variables...
hack/run-in-gopath.sh \
    go run test/typecheck/main.go "$@" --serial="${TYPECHECK_SERIAL}" --platform "${TEST_PLATFORMS}" test/e2e \
    || ret=$?

if [[ $ret -ne 0 ]]; then
  echo "!!! Type Check has failed. This may cause cross platform build failures." >&2
  echo "!!! Please see https://git.k8s.io/kubernetes/test/typecheck for more information." >&2
  exit 1
fi
