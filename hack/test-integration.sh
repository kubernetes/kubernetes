#!/bin/bash

# Copyright 2014 The Kubernetes Authors All rights reserved.
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

# any command line arguments will be passed to hack/build_go.sh to build the
# cmd/integration binary.  --use_go_build is a legitimate argument, as are
# any other build time arguments.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"
# Comma separated list of API Versions that should be tested.
KUBE_TEST_API_VERSIONS=${KUBE_TEST_API_VERSIONS:-"v1beta1,v1beta3"}
KUBE_INTEGRATION_TEST_MAX_CONCURRENCY=${KUBE_INTEGRATION_TEST_MAX_CONCURRENCY:-"-1"}
LOG_LEVEL=${LOG_LEVEL:-2}

cleanup() {
  kube::etcd::cleanup
  kube::log::status "Integration test cleanup complete"
}

runTests() {
  kube::etcd::start

  kube::log::status "Running integration test cases"
  KUBE_GOFLAGS="-tags 'integration no-docker' " \
    KUBE_RACE="-race" \
    KUBE_TEST_API_VERSIONS="$1" \
    "${KUBE_ROOT}/hack/test-go.sh" test/integration

  kube::log::status "Running integration test scenario"

  "${KUBE_OUTPUT_HOSTBIN}/integration" --v=${LOG_LEVEL} --api-version="$1" \
  --max-concurrency="${KUBE_INTEGRATION_TEST_MAX_CONCURRENCY}"

  cleanup
}

"${KUBE_ROOT}/hack/build-go.sh" "$@" cmd/integration

# Run cleanup to stop etcd on interrupt or other kill signal.
trap cleanup EXIT

# Convert the CSV to an array of API versions to test
IFS=',' read -a apiVersions <<< "${KUBE_TEST_API_VERSIONS}"
for apiVersion in "${apiVersions[@]}"; do
  runTests "${apiVersion}"
done
