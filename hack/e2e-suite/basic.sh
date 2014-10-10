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

# Launches an nginx container and verifies it can be reached. Assumes that
# we're being called by hack/e2e-test.sh (we use some env vars it sets up).

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
source "${KUBE_ROOT}/cluster/kube-env.sh"
source "${KUBE_ROOT}/cluster/$KUBERNETES_PROVIDER/util.sh"

# Launch a container
$KUBECFG -p 8080:80 run dockerfile/nginx 2 myNginx

function remove-quotes() {
  local in=$1
  stripped="${in%\"}"
  stripped="${stripped#\"}"
  echo $stripped
}

function teardown() {
  echo "Cleaning up test artifacts"
  $KUBECFG stop myNginx
  $KUBECFG rm myNginx
}

trap "teardown" EXIT

pod_id_list=$($KUBECFG '-template={{range.Items}}{{.ID}} {{end}}' -l replicationController=myNginx list pods)
# Container turn up on a clean cluster can take a while for the docker image pull.
all_running=0
while [[ $all_running -ne 1 ]]; do
  echo "Waiting for all containers in pod to come up."
  sleep 5
  all_running=1
  for id in $pod_id_list; do
    current_status=$($KUBECFG -template '{{and .CurrentState.Info.mynginx.State.Running .CurrentState.Info.net.State.Running}}' get pods/$id) || true
    if [[ "$current_status" != "{0001-01-01 00:00:00 +0000 UTC}" ]]; then
      all_running=0
    fi
  done
done

# Get minion IP addresses
detect-minions

# let images stabilize
echo "Letting images stabilize"
sleep 5

# Verify that something is listening (nginx should give us a 404)
for (( i=0; i<${#KUBE_MINION_IP_ADDRESSES[@]}; i++)); do
  ip_address=${KUBE_MINION_IP_ADDRESSES[$i]}
  echo "Trying to reach nginx instance that should be running at ${ip_address}:8080..."
  curl "http://${ip_address}:8080"
done

exit 0
