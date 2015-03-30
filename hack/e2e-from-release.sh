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

# This script will build a lmktfy.tar.gz and
# lmktfy-test.tar.gz, unpack them in a separate directory and try
# to run e2es from there, as if it were a binary distribution separate
# from the git tree.

set -o errexit
set -o nounset
set -o pipefail

LMKTFY_ROOT=$(dirname "${BASH_SOURCE}")/..

# Build a release
echo
echo "... building release"
echo
"${LMKTFY_ROOT}/build/release.sh"

echo
echo "... ensuring git e2e cluster is down"
echo
go run "${LMKTFY_ROOT}/hack/e2e.go" -v -down

TEST_DEPLOY="${LMKTFY_ROOT}/_output/test-deploy"
echo
echo "... deploying lmktfy.tar.gz / lmktfy-test.tar.gz to ${TEST_DEPLOY}"
echo
rm -rf ${TEST_DEPLOY}
mkdir -p ${TEST_DEPLOY}
tar -C "${TEST_DEPLOY}" -xzf _output/release-tars/lmktfy.tar.gz
tar -C "${TEST_DEPLOY}" -xzf _output/release-tars/lmktfy-test.tar.gz

# Nothing past here should touch the git checkout.
unset LMKTFY_ROOT
cd "${TEST_DEPLOY}/lmktfy"

echo
echo "... running e2e tests from ${TEST_DEPLOY}"
echo
go run ./hack/e2e.go -v -up -test -down "$*" || (
  status=$?

  echo
  echo "... e2e tests failed from ${TEST_DEPLOY}! To reproduce this command:"
  echo "...   (cd '${TEST_DEPLOY}/lmktfy' && go run ./hack/e2e.go -v -up -test -down" "$*" ")"
  echo

  exit ${status})

exit 0
