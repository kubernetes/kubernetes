#!/bin/bash

# Copyright 2014 The Kubernetes Authors All rights reserved.
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

# Validates that the cluster is healthy.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/cluster/kube-env.sh"
source "${KUBE_ROOT}/cluster/kube-util.sh"

MINIONS_FILE=/tmp/minions-$$
trap 'rm -rf "${MINIONS_FILE}"' EXIT

# Make several attempts to deal with slow cluster birth.
attempt=0
while true; do
  # The "kubectl get nodes -o template" exports node information.
  #
  # Echo the output and gather 2 counts:
  #  - Total number of nodes.
  #  - Number of "ready" nodes.
  "${KUBE_ROOT}/cluster/kubectl.sh" get nodes > "${MINIONS_FILE}" || true
  found=$(cat "${MINIONS_FILE}" | sed '1d' | grep -c .) || true
  ready=$(cat "${MINIONS_FILE}" | sed '1d' | awk '{print $NF}' | grep -c '^Ready') || true

  if (( ${found} == "${NUM_MINIONS}" )) && (( ${ready} == "${NUM_MINIONS}")); then
    break
  else
    # Set the timeout to ~10minutes (40 x 15 second) to avoid timeouts for 100-node clusters.
    if (( attempt > 40 )); then
      echo -e "${color_red}Detected ${ready} ready nodes, found ${found} nodes out of expected ${NUM_MINIONS}. Your cluster may not be working.${color_norm}"
      cat -n "${MINIONS_FILE}"
      exit 2
		else
      echo -e "${color_yellow}Waiting for ${NUM_MINIONS} ready nodes. ${ready} ready nodes, ${found} registered. Retrying.${color_norm}"
    fi
    attempt=$((attempt+1))
    sleep 15
  fi
done
echo "Found ${found} node(s)."
"${KUBE_ROOT}/cluster/kubectl.sh" get nodes

attempt=0
while true; do
  # The "kubectl componentstatuses -o template" exports components health information.
  #
  # Echo the output and gather 2 counts:
  #  - Total number of componentstatuses.
  #  - Number of "healthy" components.
  cs_status=$("${KUBE_ROOT}/cluster/kubectl.sh" get componentstatuses -o template --template='{{range .items}}{{with index .conditions 0}}{{.type}}:{{.status}},{{end}}{{end}}' --api-version=v1) || true
  componentstatuses=$(echo "${cs_status}" | tr "," "\n" | grep -c 'Healthy:') || true
  healthy=$(echo "${cs_status}" | tr "," "\n" | grep -c 'Healthy:True') || true

  if ((componentstatuses > healthy)); then
    if ((attempt < 5)); then
      echo -e "${color_yellow}Cluster not working yet.${color_norm}"
      attempt=$((attempt+1))
      sleep 30
    else
      echo -e " ${color_yellow}Validate output:${color_norm}"
      "${KUBE_ROOT}/cluster/kubectl.sh" get cs
      echo -e "${color_red}Validation returned one or more failed components. Cluster is probably broken.${color_norm}"
      exit 1
    fi
  else
    break
  fi
done

echo "Validate output:"
"${KUBE_ROOT}/cluster/kubectl.sh" get cs
echo -e "${color_green}Cluster validation succeeded${color_norm}"
