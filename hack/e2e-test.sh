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

# Starts a Kubernetes cluster, runs the e2e test suite, and shuts it
# down.

source $(dirname $0)/../cluster/kube-env.sh
source $(dirname $0)/../cluster/$KUBERNETES_PROVIDER/util.sh

# For debugging of this test's components, it's helpful to leave the test
# cluster running.
ALREADY_UP=${1:-0}
LEAVE_UP=${2:-0}
TEAR_DOWN=${3:-0}

# Exit on error
set -e

# Use testing config
export KUBE_CONFIG_FILE="config-test.sh"
export KUBE_REPO_ROOT="$(dirname $0)/.."
export CLOUDCFG="${KUBE_REPO_ROOT}/cluster/kubecfg.sh -expect_version_match"

if [[ $TEAR_DOWN -ne 0 ]]; then
  detect-project
  trap test-teardown EXIT
  exit 0
fi

# Build a release required by the test provider [if any]
test-build-release

if [[ ${ALREADY_UP} -ne 1 ]]; then
  # Now bring a test cluster up with that release.
  $(dirname $0)/../cluster/kube-up.sh
else
  # Just push instead
  $(dirname $0)/../cluster/kube-push.sh
fi

# Perform any required setup of the cluster
test-setup

set +e

if [[ ${LEAVE_UP} -ne 1 ]]; then
  trap test-teardown EXIT
fi

any_failed=0
for test_file in $(ls $(dirname $0)/e2e-suite/); do
  "$(dirname $0)/e2e-suite/${test_file}"
  result="$?"
  if [[ "${result}" -eq "0" ]]; then
    echo "${test_file} returned ${result}; passed!"
  else
    echo "${test_file} returned ${result}; FAIL!"
    any_failed=1
  fi
done

if [[ ${any_failed} -ne 0 ]]; then
  echo "At least one test failed."
fi

exit ${any_failed}
