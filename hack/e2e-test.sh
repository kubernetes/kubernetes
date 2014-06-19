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

# For debugging of this test's components, it's helpful to leave the test
# cluster running.
ALREADY_UP=${1:-0}
LEAVE_UP=${2:-0}

# Exit on error
set -e

HAVE_JQ=$(which jq)
if [[ -z ${HAVE_JQ} ]]; then
  echo "Please install jq, e.g.: 'sudo apt-get install jq' or, "
  echo "if you're on a mac with homebrew, 'brew install jq'."
  exit 1
fi

# Use testing config
export KUBE_CONFIG_FILE="config-test.sh"
export KUBE_REPO_ROOT="$(dirname $0)/.."
export CLOUDCFG="${KUBE_REPO_ROOT}/cluster/cloudcfg.sh"

source "${KUBE_REPO_ROOT}/cluster/util.sh"
${KUBE_REPO_ROOT}/hack/build-go.sh

# Build a release
$(dirname $0)/../release/release.sh

if [[ ${ALREADY_UP} -ne 1 ]]; then
  # Now bring a test cluster up with that release.
  $(dirname $0)/../cluster/kube-up.sh
else
  # Just push instead
  $(dirname $0)/../cluster/kube-push.sh
fi

# Detect the project into $PROJECT if it isn't set
detect-project

set +e

if [[ ${ALREADY_UP} -ne 1 ]]; then
  # Open up port 80 & 8080 so common containers on minions can be reached
  gcutil addfirewall \
    --norespect_terminal_width \
    --project ${PROJECT} \
    --target_tags ${MINION_TAG} \
    --allowed tcp:80,tcp:8080 \
    --network ${NETWORK} \
    ${MINION_TAG}-http-alt
fi

# Auto shutdown cluster when we exit
function shutdown-test-cluster () {
  echo "Shutting down test cluster in background."
  gcutil deletefirewall  \
    --project ${PROJECT} \
    --norespect_terminal_width \
    --force \
    ${MINION_TAG}-http-alt &
  $(dirname $0)/../cluster/kube-down.sh > /dev/null &
}

if [[ ${LEAVE_UP} -ne 1 ]]; then
  trap shutdown-test-cluster EXIT
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
