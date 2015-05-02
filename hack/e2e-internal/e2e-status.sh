#!/bin/bash

# Copyright 2015 The Kubernetes Authors All rights reserved.
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

# e2e-status checks that the status of a cluster is acceptable for running
# e2e tests.
set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..

: ${KUBE_VERSION_ROOT:=${KUBE_ROOT}}
: ${KUBECTL:="${KUBE_VERSION_ROOT}/cluster/kubectl.sh"}
: ${KUBE_CONFIG_FILE:="config-test.sh"}

export KUBECTL KUBE_CONFIG_FILE

source "${KUBE_ROOT}/cluster/kube-env.sh"
source "${KUBE_VERSION_ROOT}/cluster/${KUBERNETES_PROVIDER}/util.sh"

prepare-e2e

${KUBECTL} version

# Before running tests, ensure that all pods are 'Running'. Tests can timeout
# and fail because the test pods don't run in time. The problem is that the pods
# that a cluster runs on startup take too long to start running, with sequential
# Docker pulls of large images being the culprit. These startup pods block the
# test pods from running.

# Settings:
# timeout is in seconds; 1200 = 20 minutes.
timeout=1200
# pause is how many seconds to sleep between pod get calls.
pause=5
# min_pods is the minimum number of pods we require.
min_pods=1

# Check pod statuses.
deadline=$(($(date '+%s')+${timeout}))
echo "Waiting at most ${timeout} seconds for all pods to be 'Running'" >&2
all_running=0
until [[ ${all_running} == 1 ]]; do
  if [[ "$(date '+%s')" -ge "${deadline}" ]]; then
    echo "All pods never 'Running' in time." >&2
    exit 1
  fi
  statuses=($(${KUBECTL} get pods --template='{{range.items}}{{.status.phase}} {{end}}' --api-version=v1beta3))

  # Ensure that we have enough pods.
  echo "Found ${#statuses[@]} pods with statuses: ${statuses[@]}" >&2
  if [[ ${#statuses[@]} -lt ${min_pods} ]]; then
    continue
  fi

  # Then, ensure all pods found are 'Running'.
  found_running=1
  for status in "${statuses[@]}"; do
    if [[ "${status}" != "Running" ]]; then
      # If we find a pod that isn't 'Running', sleep here to avoid delaying
      # other code paths (where all pods are 'Running').
      found_running=0
      sleep ${pause}
      break
    fi
  done
  all_running=${found_running}
done
echo "All pods are 'Running'" >&2
