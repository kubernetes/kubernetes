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

# Exit on error
set -e

# Use testing config
export KUBE_CONFIG_FILE="config-test.sh"
export KUBE_REPO_ROOT="$(dirname $0)/.."
export CLOUDCFG="${KUBE_REPO_ROOT}/cluster/cloudcfg.sh"

source "${KUBE_REPO_ROOT}/cluster/util.sh"

# Build a release
$(dirname $0)/../release/release.sh

# Now bring a test cluster up with that release.
$(dirname $0)/../cluster/kube-up.sh

# Detect the project into $PROJECT if it isn't set
detect-project

set +e

# Open up port 80 & 8080 so common containers on minions can be reached
gcutil addfirewall \
  --norespect_terminal_width \
  --project ${PROJECT} \
  --target_tags ${MINION_TAG} \
  --allowed tcp:80 \
  --allowed tcp:8080 \
  --network ${NETWORK} \
  ${MINION_TAG}-http-alt

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
trap shutdown-test-cluster EXIT

any_failed=0
for test_file in "$(dirname $0)/e2e-suite/*.sh"; do
  $test_file
  if [[ -z $? ]]; then
    echo "${test_file}: passed!"
  else
    echo "${test_file}: FAILED!"
    any_failed=1
  fi
done

exit ${any_failed}
