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

# Launches a container and verifies it can be reached. Assumes that
# we're being called by hack/e2e-test.sh (we use some env vars it sets up).

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
source "${KUBE_ROOT}/cluster/kube-env.sh"
source "${KUBE_ROOT}/cluster/$KUBERNETES_PROVIDER/util.sh"

function teardown() {
  echo "Cleaning up test artifacts"
  ${KUBECFG} delete pods/liveness-http
}

function waitForNotPending() {
  pod_id_list=$($KUBECFG '-template={{range.items}}{{.id}} {{end}}' -l test=liveness list pods)
  # Pod turn up on a clean cluster can take a while for the docker image pull.
  all_running=0
  for i in $(seq 1 24); do
    echo "Waiting for pod to come up."
    sleep 5
    all_running=1
    for id in $pod_id_list; do
      current_status=$($KUBECFG -template '{{.currentState.status}}' get pods/$id) || true
      if [[ "$current_status" == "Pending" ]]; then
        all_running=0
        break
      fi
    done
    if [[ "${all_running}" == 1 ]]; then
      break
    fi
  done
  if [[ "${all_running}" == 0 ]]; then
    echo "Pods did not come up in time"
    exit 1
  fi
}

trap "teardown" EXIT

${KUBECFG} -c ${KUBE_ROOT}/examples/liveness/http-liveness.yaml create pods
waitForNotPending

before=$(${KUBECFG} '-template={{.currentState.info.liveness.restartCount}}' get pods/liveness-http)

echo "Waiting for restarts."
for i in $(seq 1 24); do
  sleep 10 
  after=$(${KUBECFG} '-template={{.currentState.info.liveness.restartCount}}' get pods/liveness-http)
  echo "Restarts: ${after} > ${before}"
  if [[ "${after}" > "${before}" ]]; then
    break
  fi
done

if [[ "${before}" < "${after}" ]]; then
  exit 0
fi

echo "Unexpected absence of failures."
echo "Restarts before: ${before}."
echo "Restarts after: ${after}"
exit 1
