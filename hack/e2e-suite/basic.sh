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
  $KUBECFG stop my-hostname
  $KUBECFG rm my-hostname
}

trap "teardown" EXIT

# Determine which pod image to launch (e.g. private.sh launches a different one).
pod_img_srv="${POD_IMG_SRV:-kubernetes/serve_hostname:1.1}"

# Launch some pods.
num_pods=2
$KUBECFG -p 8080:9376 run "${pod_img_srv}" ${num_pods} my-hostname

# List the pods.
pod_id_list=$($KUBECFG '-template={{range.items}}{{.id}} {{end}}' -l name=my-hostname list pods)
echo "pod_id_list: ${pod_id_list}"
if [[ -z "${pod_id_list:-}" ]]; then
  echo "Pod ID list is empty. It should have a set of pods to verify."
  exit 1
fi

# Pod turn up on a clean cluster can take a while for the docker image pull.
all_running=0
for i in $(seq 1 24); do
  echo "Waiting for pods to come up."
  sleep 5
  all_running=1
  for id in $pod_id_list; do
    current_status=$($KUBECFG '-template={{.currentState.status}}' get pods/$id) || true
    if [[ "$current_status" != "Running" ]]; then
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

# let images stabilize
echo "Letting images stabilize"
sleep 5

# Verify that something is listening.
for id in ${pod_id_list}; do
  ip=$($KUBECFG '-template={{.currentState.hostIP}}' get pods/$id)
  echo "Trying to reach server that should be running at ${ip}:8080..."
  server_running=0
  for i in $(seq 1 5); do
    echo "--- trial ${i}"
    output=$(curl -s -connect-timeout 1 "http://${ip}:8080" || true)
    if echo $output | grep "${id}" &> /dev/null; then
      server_running=1
      break
    fi
    sleep 2
  done
  if [[ "${server_running}" -ne 1 ]]; then
    echo "Server never running at ${ip}:8080..."
    exit 1
  fi
done

exit 0
