#!/bin/bash

# Copyright 2014 Google Inc. All rights reserved.
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

LMKTFY_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${LMKTFY_ROOT}/hack/lib/init.sh"
# Comma separated list of API Versions that should be tested.
LMKTFY_TEST_API_VERSIONS=${LMKTFY_TEST_API_VERSIONS:-"v1beta1,v1beta3"}


cleanup() {
  lmktfy::etcd::cleanup
  lmktfy::log::status "Integration test cleanup complete"
}

runTests() {
  lmktfy::etcd::start

  lmktfy::log::status "Running integration test cases"
  LMKTFY_GOFLAGS="-tags 'integration no-docker' " \
    LMKTFY_RACE="-race" \
    LMKTFY_TEST_API_VERSIONS="$1" \
    "${LMKTFY_ROOT}/hack/test-go.sh" test/integration

  lmktfy::log::status "Running integration test scenario"

  "${LMKTFY_OUTPUT_HOSTBIN}/integration" --v=2 --apiVersion="$1"

  cleanup
}

"${LMKTFY_ROOT}/hack/build-go.sh" "$@" cmd/integration

# Run cleanup to stop etcd on interrupt or other kill signal.
trap cleanup EXIT

# Convert the CSV to an array of API versions to test
IFS=',' read -a apiVersions <<< "${LMKTFY_TEST_API_VERSIONS}"
for apiVersion in "${apiVersions[@]}"; do
  runTests "${apiVersion}"
done
