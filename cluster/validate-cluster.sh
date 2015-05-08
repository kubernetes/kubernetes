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
source "${KUBE_ROOT}/cluster/${KUBERNETES_PROVIDER}/util.sh"

MINIONS_FILE=/tmp/minions-$$
trap 'rm -rf "${MINIONS_FILE}"' EXIT

# Make several attempts to deal with slow cluster birth.
attempt=0
while true; do
  # The "kubectl get nodes" output is three columns like this:
  #
  #     NAME                     LABELS    STATUS
  #     kubernetes-minion-03nb   <none>    Ready
  #
  # Echo the output, strip the first line, then gather 2 counts:
  #  - Total number of nodes.
  #  - Number of "ready" nodes.
  "${KUBE_ROOT}/cluster/kubectl.sh" get nodes > "${MINIONS_FILE}" || true
  found=$(cat "${MINIONS_FILE}" | sed '1d' | grep -c .) || true
  ready=$(cat "${MINIONS_FILE}" | sed '1d' | awk '{print $NF}' | grep -c '^Ready') || true

  if (( ${found} == "${NUM_MINIONS}" )) && (( ${ready} == "${NUM_MINIONS}")); then
    break
  else
    if (( attempt > 5 )); then
      echo -e "${color_red}Detected ${ready} ready nodes, found ${found} nodes out of expected ${NUM_MINIONS}. Your cluster may not be working. ${color_norm}"
      cat -n "${MINIONS_FILE}"
      exit 2
    fi
    attempt=$((attempt+1))
    sleep 30
  fi
done
echo "Found ${found} nodes."
cat -n "${MINIONS_FILE}"

attempt=0
while true; do
  kubectl_output=$("${KUBE_ROOT}/cluster/kubectl.sh" get cs) || true

  # The "kubectl componentstatuses" output is four columns like this:
  #
  #     COMPONENT            HEALTH    MSG       ERR
  #     controller-manager   Healthy   ok        nil
  #
  # Parse the output to capture the value of the second column("HEALTH"), then use grep to
  # count the number of times it doesn't match "Healthy".
  non_success_count=$(echo "${kubectl_output}" | \
    sed '1d' |
    sed -n 's/^[[:alnum:][:punct:]]/&/p' | \
    grep --invert-match -c '^[[:alnum:][:punct:]]\{1,\}[[:space:]]\{1,\}Healthy') || true

  if ((non_success_count > 0)); then
    if ((attempt < 5)); then
      echo -e "${color_yellow}Cluster not working yet.${color_norm}"
      attempt=$((attempt+1))
      sleep 30
    else
      echo -e " ${color_yellow}Validate output:${color_norm}"
      echo "${kubectl_output}"
      echo -e "${color_red}Validation returned one or more failed components. Cluster is probably broken.${color_norm}"
      exit 1
    fi
  else
    break
  fi
done

echo "Validate output:"
echo "${kubectl_output}"
echo -e "${color_green}Cluster validation succeeded${color_norm}"
