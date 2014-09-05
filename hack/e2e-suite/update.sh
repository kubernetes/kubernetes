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
set -x

source "${KUBE_REPO_ROOT}/cluster/kube-env.sh"
source "${KUBE_REPO_ROOT}/cluster/$KUBERNETES_PROVIDER/util.sh"


CONTROLLER_NAME=update-demo

function validate() {
  NUM_REPLICAS=$1
  CONTAINER_IMAGE_VERSION=$2
  POD_ID_LIST=$($KUBECFG '-template={{range.Items}}{{.ID}} {{end}}' -l replicationController=${CONTROLLER_NAME} list pods)
  # Container turn up on a clean cluster can take a while for the docker image pull.
  ALL_RUNNING=0
  while [ $ALL_RUNNING -ne 1 ]; do
    echo "Waiting for all containers in pod to come up."
    sleep 5
    ALL_RUNNING=1
    for id in $POD_ID_LIST; do
      TEMPLATE_STRING="{{and ((index .CurrentState.Info \"${CONTROLLER_NAME}\").State.Running) .CurrentState.Info.net.State.Running}}"
      CURRENT_STATUS=$($KUBECFG -template "${TEMPLATE_STRING}" get pods/$id)
      if [ "$CURRENT_STATUS" != "true" ]; then
        ALL_RUNNING=0
      else
        CURRENT_IMAGE=$($KUBECFG -template "{{(index .CurrentState.Info \"${CONTROLLER_NAME}\").Config.Image}}" get pods/$id)
        if [ "$CURRENT_IMAGE" != "${DOCKER_HUB_USER}/update-demo:${CONTAINER_IMAGE_VERSION}" ]; then
          ALL_RUNNING=0
        fi
      fi
    done
  done

  ids=($POD_ID_LIST)
  if [ ${#ids[@]} -ne $NUM_REPLICAS ]; then
    echo "Unexpected number of pods: ${#ids[@]}.  Expected $NUM_REPLICAS"
    exit 1
  fi
}

export DOCKER_HUB_USER=jbeda

# Launch a container
${KUBE_REPO_ROOT}/examples/update-demo/1-create-replication-controller.sh

function teardown() {
  echo "Cleaning up test artifacts"
  ${KUBE_REPO_ROOT}/examples/update-demo/4-down.sh
}

trap "teardown" EXIT

validate 2 nautilus

${KUBE_REPO_ROOT}/examples/update-demo/2-scale.sh 1
sleep 2
validate 1 nautilus

${KUBE_REPO_ROOT}/examples/update-demo/2-scale.sh 2
sleep 2
validate 2 nautilus

${KUBE_REPO_ROOT}/examples/update-demo/3-rolling-update.sh kitten 1s
sleep 2
validate 2 kitten

exit 0
