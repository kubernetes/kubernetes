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

EXPECTED_NUM_NODES="${NUM_MINIONS}"
if [[ "${REGISTER_MASTER_KUBELET:-}" == "true" ]]; then
  EXPECTED_NUM_NODES=$((EXPECTED_NUM_NODES+1))
fi
# Make several attempts to deal with slow cluster birth.
attempt=0
while true; do
  # The "kubectl get nodes -o template" exports node information.
  #
  # Echo the output and gather 2 counts:
  #  - Total number of nodes.
  #  - Number of "ready" nodes.
  #
  # Suppress errors from kubectl output because during cluster bootstrapping
  # for clusters where the master node is registered, the apiserver will become
  # available and then get restarted as the kubelet configures the docker bridge.
  nodes_status=$("${KUBE_ROOT}/cluster/kubectl.sh" get nodes -o template --template='{{range .items}}{{with index .status.conditions 0}}{{.type}}:{{.status}},{{end}}{{end}}' --api-version=v1) || true
  found=$(echo "${nodes_status}" | tr "," "\n" | grep -c 'Ready:') || true
  ready=$(echo "${nodes_status}" | tr "," "\n" | grep -c 'Ready:True') || true

  if (( "${found}" == "${EXPECTED_NUM_NODES}" )) && (( "${ready}" == "${EXPECTED_NUM_NODES}")); then
    break
  elif (( "${found}" > "${EXPECTED_NUM_NODES}" )) && (( "${ready}" > "${EXPECTED_NUM_NODES}")); then
    echo -e "${color_red}Detected ${ready} ready nodes, found ${found} nodes out of expected ${EXPECTED_NUM_NODES}. Found more nodes than expected, your cluster may not behave correctly.${color_norm}"
    break
  else
    # Set the timeout to ~10minutes (40 x 15 second) to avoid timeouts for 100-node clusters.
    if (( attempt > 40 )); then
      echo -e "${color_red}Detected ${ready} ready nodes, found ${found} nodes out of expected ${EXPECTED_NUM_NODES}. Your cluster may not be working.${color_norm}"
      "${KUBE_ROOT}/cluster/kubectl.sh" get nodes
      exit 2
		else
      echo -e "${color_yellow}Waiting for ${EXPECTED_NUM_NODES} ready nodes. ${ready} ready nodes, ${found} registered. Retrying.${color_norm}"
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
