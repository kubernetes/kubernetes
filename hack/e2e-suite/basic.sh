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

# Exit on error
set -e

source "${KUBE_REPO_ROOT}/cluster/util.sh"
detect-project

# Launch a container
$CLOUDCFG -p 8080:80 run dockerfile/nginx 2 myNginx

function remove-quotes() {
  local in=$1
  stripped="${in%\"}"
  stripped="${stripped#\"}"
  echo $stripped
}

POD_ID_LIST=$($CLOUDCFG -json -l name=myNginx list pods | jq ".items[].id")
# Container turn up on a clean cluster can take a while for the docker image pull.
ALL_RUNNING=false
while [[ $ALL_RUNNING -ne "true" ]]; do
  echo "Waiting for containers to come up."
  sleep 5
  ALL_RUNNING=true
  for id in $POD_ID_LIST; do
    CURRENT_STATUS=$(remove-quotes $($CLOUDCFG -json get "pods/$(remove-quotes ${id})" | jq '.currentState.status'))
    if [[ $CURRENT_STATUS -ne "Running" ]]; then
      ALL_RUNNING=false
    fi
  done
done

# Get minion IP addresses
detect-minions

# Verify that something is listening (nginx should give us a 404)
for (( i=0; i<${#KUBE_MINION_IP_ADDRESSES[@]}; i++)); do
  IP_ADDRESS=${KUBE_MINION_IP_ADDRESSES[$i]}
  echo "Trying to reach nginx instance that should be running at ${IP_ADDRESS}:8080..."
  curl "http://${IP_ADDRESS}:8080"
done

$CLOUDCFG stop myNginx
$CLOUDCFG rm myNginx

exit 0
