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

MINIONS_FILE=/tmp/minions
"${KUBE_ROOT}/cluster/kubecfg.sh" -template $'{{range.items}}{{.id}}\n{{end}}' list minions > ${MINIONS_FILE}

# On vSphere, use minion IPs as their names
if [[ "${KUBERNETES_PROVIDER}" == "vsphere" ]]; then
  for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
    MINION_NAMES[i]=${KUBE_MINION_IP_ADDRESSES[i]}
  done
fi

for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
    # Grep returns an exit status of 1 when line is not found, so we need the : to always return a 0 exit status
    count=$(grep -c ${MINION_NAMES[i]} ${MINIONS_FILE}) || :
    if [[ "$count" == "0" ]]; then
        echo "Failed to find ${MINION_NAMES[$i]}, cluster is probably broken."
        exit 1
    fi

    NAME=${MINION_NAMES[i]}
    if [ "$KUBERNETES_PROVIDER" != "vsphere" ]; then
      # Grab fully qualified name
      NAME=$(grep "${MINION_NAMES[i]}" ${MINIONS_FILE})
    fi

    # Make sure the kubelet is healthy
    curl_output=$(curl -s --insecure --user "${KUBE_USER}:${KUBE_PASSWORD}" \
        "https://${KUBE_MASTER_IP}/api/v1beta1/proxy/minions/${NAME}/healthz")
    if [[ "${curl_output}" != "ok" ]]; then
        echo "Kubelet failed to install on ${MINION_NAMES[$i]}. Your cluster is unlikely to work correctly."
        echo "Please run ./cluster/kube-down.sh and re-create the cluster. (sorry!)"
        exit 1
    else
        echo "Kubelet is successfully installed on ${MINION_NAMES[$i]}"
    fi
done
echo "Cluster validation succeeded"
