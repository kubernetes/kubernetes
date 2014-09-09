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

# exit on any error
set -e

source $(dirname $0)/../kube-env.sh
source $(dirname $0)/../$KUBERNETES_PROVIDER/util.sh

get-password
detect-master > /dev/null
detect-minions > /dev/null

MINIONS_FILE=/tmp/minions
$(dirname $0)/../kubecfg.sh -template '{{range.Items}}{{.ID}}:{{end}}' list minions > ${MINIONS_FILE}

for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
    # Grep returns an exit status of 1 when line is not found, so we need the : to always return a 0 exit status
    count=$(grep -c ${MINION_NAMES[i]} ${MINIONS_FILE}) || :
    if [ "$count" == "0" ]; then
        echo "Failed to find ${MINION_NAMES[i]}, cluster is probably broken."
        exit 1
    fi

    # Make sure the kubelet is healthy
    if [ "$(curl --insecure --user ${user}:${passwd} https://${KUBE_MASTER_IP}/proxy/minion/${MINION_NAMES[$i]}/healthz)" != "ok" ]; then
        echo "Kubelet failed to install on ${MINION_NAMES[$i]}. Your cluster is unlikely to work correctly."
        echo "Please run ./cluster/kube-down.sh and re-create the cluster. (sorry!)"
        exit 1
    else
        echo "Kubelet is successfully installed on ${MINION_NAMES[$i]}"
    fi
done
echo "Cluster validation succeeded"
