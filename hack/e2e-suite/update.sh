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


CONTROLLER_NAME=update-demo

function validate() {
  local num_replicas=$1
  local container_image_version=$2

  # Container turn up on a clean cluster can take a while for the docker image pull.
  local num_running=0
  local i=0
  while [[ ${num_running} -ne ${num_replicas} && ${i} -ne 100]]; do
    ((i++)) || true
    echo "Waiting for all containers in pod to come up. Currently: ${num_running}/${num_replicas}"
    sleep 2

    local pod_id_list
    pod_id_list=($($KUBECFG -template='{{range.items}}{{.id}} {{end}}' -l name="${CONTROLLER_NAME}" list pods))

    echo "  ${#pod_id_list[@]} out of ${num_replicas} created"

    local id
    num_running=0
    if [[ ${#pod_id_list[@]} -ne $num_replicas ]]; then
      echo "Too few or too many replicas."
      continue
    fi
    for id in "${pod_id_list[@]+${pod_id_list[@]}}"; do
      local template_string current_status current_image host_ip

      # NB: kubectl & kubecfg add the "exists" function to the standard template functions.
      # This lets us check to see if the "running" entry exists for each of the containers
      # we care about. Exists will never return an error and it's safe to check a chain of
      # things, any one of which may not exist. In the below template, all of info, 
      # containername, and running might be nil, so the normal index function isn't very
      # helpful.
      # This template is unit-tested in kubec{tl|fg}, so if you change it, update the unit test.
      #
      # You can read about the syntax here: http://golang.org/pkg/text/template/
      template_string="{{and (exists . \"currentState\" \"info\" \"${CONTROLLER_NAME}\" \"state\" \"running\") (exists . \"currentState\" \"info\" \"POD\" \"state\" \"running\")}}"
      current_status=$($KUBECFG -template="${template_string}" get "pods/$id") || {
        if [[ $current_status =~ "pod \"${id}\" not found" ]]; then
          echo "  $id no longer exists"
          continue
        else
          echo "  kubecfg failed with error:"
          echo $current_status
          exit -1
        fi
      }
      if [[ "$current_status" == "false" ]]; then
        echo "  $id is created but not running."
        continue
      fi

      echo "  $id is created and both POD and update-demo containers are running: $current_status"

      template_string="{{(index .currentState.info \"${CONTROLLER_NAME}\").image}}"
      current_image=$($KUBECFG -template="${template_string}" get "pods/$id") || true
      if [[ "$current_image" != "${DOCKER_HUB_USER}/update-demo:${container_image_version}" ]]; then
        echo "  ${id} is created but running wrong image"
        continue
      fi


      host_ip=$($KUBECFG -template='{{.currentState.hostIP}}' get pods/$id)
      curl -s --max-time 5 --fail http://${host_ip}:8080/data.json \
          | grep -q ${container_image_version} || {
        echo "  ${id} is running the right image but curl to contents failed or returned wrong info"
        continue

      }

      echo "  ${id} is verified up and running"

      ((num_running++)) || true
    done
  done
  return 0
}

export DOCKER_HUB_USER=davidopp

# Launch a container
${KUBE_ROOT}/examples/update-demo/2-create-replication-controller.sh

function teardown() {
  echo "Cleaning up test artifacts"
  ${KUBE_ROOT}/examples/update-demo/5-down.sh
}

trap "teardown" EXIT

validate 2 nautilus

${KUBE_ROOT}/examples/update-demo/3-scale.sh 1
sleep 2
validate 1 nautilus

${KUBE_ROOT}/examples/update-demo/3-scale.sh 2
sleep 2
validate 2 nautilus

${KUBE_ROOT}/examples/update-demo/4-rolling-update.sh kitten 1s
sleep 2
validate 2 kitten

exit 0
