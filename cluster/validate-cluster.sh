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

# Bring up a Kubernetes cluster.
#
# If the full release name (gs://<bucket>/<release>) is passed in then we take
# that directly.  If not then we assume we are doing development stuff and take
# the defaults in the release config.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/cluster/kube-env.sh"
source "${KUBE_ROOT}/cluster/${KUBERNETES_PROVIDER}/util.sh"

get-password
detect-master > /dev/null
detect-minions > /dev/null

MINIONS_FILE=/tmp/minions-$$
trap 'rm -rf "${MINIONS_FILE}"' EXIT
# Make several attempts to deal with slow cluster birth.
attempt=0
while true; do
  "${KUBE_ROOT}/cluster/kubectl.sh" get nodes -o template -t $'{{range.items}}{{.metadata.name}}\n{{end}}' --api-version=v1beta3 > "${MINIONS_FILE}"
  found=$(grep -c . "${MINIONS_FILE}")
  if [[ ${found} == "${NUM_MINIONS}" ]]; then
    break
  else
    if (( attempt > 5 )); then
      echo -e "${color_red}Detected ${found} nodes out of ${NUM_MINIONS}. Your cluster may not be working. ${color_norm}"
      cat -n "${MINIONS_FILE}"
      exit 2
    fi
    attempt=$((attempt+1))
    sleep 30
  fi
done
echo "Found ${found} nodes."
cat -n "${MINIONS_FILE}"

# On vSphere, use minion IPs as their names
if [[ "${KUBERNETES_PROVIDER}" == "vsphere" || "${KUBERNETES_PROVIDER}" == "vagrant" || "${KUBERNETES_PROVIDER}" == "libvirt-coreos" || "${KUBERNETES_PROVIDER}" == "juju" ]]  ; then
  MINION_NAMES=("${KUBE_MINION_IP_ADDRESSES[@]}")
fi

# On AWS we can't really name the minions, so just trust that if the number is right, the right names are there.
if [[ "${KUBERNETES_PROVIDER}" == "aws" ]]; then
  MINION_NAMES=("$(cat ${MINIONS_FILE})")
  # /healthz validation isn't working for some reason on AWS.  So just hope for the best.
  # TODO: figure out why and fix, it must be working in some form, or else clusters wouldn't work.
  echo "Kubelet health checking on AWS isn't currently supported, assuming everything is good..."
  echo -e "${color_green}Cluster validation succeeded${color_norm}"
  exit 0
fi

for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
    # Grep returns an exit status of 1 when line is not found, so we need the : to always return a 0 exit status
    count=$(grep -c "${MINION_NAMES[$i]}" "${MINIONS_FILE}") || :
    if [[ "${count}" == "0" ]]; then
      echo -e "${color_red}Failed to find ${MINION_NAMES[$i]}, cluster is probably broken.${color_norm}"
      cat -n "${MINIONS_FILE}"
      exit 1
    fi

    name="${MINION_NAMES[$i]}"
    if [[ "$KUBERNETES_PROVIDER" != "vsphere" &&  "$KUBERNETES_PROVIDER" != "vagrant" && "$KUBERNETES_PROVIDER" != "libvirt-coreos" && "$KUBERNETES_PROVIDER" != "juju" ]]; then
      # Grab fully qualified name
      name=$(grep "${MINION_NAMES[$i]}\." "${MINIONS_FILE}")
    fi

    # Make sure the kubelet is healthy.
    # Make several attempts to deal with slow cluster birth.
    attempt=0
    while true; do
      echo -n "Attempt $((attempt+1)) at checking Kubelet installation on node ${MINION_NAMES[$i]} ..."
      if [[ "$KUBERNETES_PROVIDER" != "libvirt-coreos" && "$KUBERNETES_PROVIDER" != "juju" ]]; then
        curl_output=$(curl -s --insecure --user "${KUBE_USER}:${KUBE_PASSWORD}" \
          "https://${KUBE_MASTER_IP}/api/v1beta1/proxy/minions/${name}/healthz")
      else
        curl_output=$(curl -s \
          "http://${KUBE_MASTER_IP}:8080/api/v1beta1/proxy/minions/${name}/healthz")
      fi
      if [[ "${curl_output}" != "ok" ]]; then
          if (( attempt > 5 )); then
            echo
            echo -e "${color_red}Kubelet failed to install on node ${MINION_NAMES[$i]}. Your cluster is unlikely to work correctly."
            echo -e "Please run ./cluster/kube-down.sh and re-create the cluster. (sorry!)${color_norm}"
            exit 1
          fi
      else
          echo -e " ${color_green}[working]${color_norm}"
          break
      fi
      echo -e " ${color_yellow}[not working yet]${color_norm}"
      attempt=$((attempt+1))
      sleep 30
    done
done
echo -e "${color_green}Cluster validation succeeded${color_norm}"
