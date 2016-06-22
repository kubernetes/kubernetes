#!/bin/bash

# Copyright 2015 The Kubernetes Authors All rights reserved.
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

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..

: ${KUBECTL:=${KUBE_ROOT}/cluster/kubectl.sh}
: ${KUBE_CONFIG_FILE:="config-test.sh"}

export KUBECTL KUBE_CONFIG_FILE

source "${KUBE_ROOT}/cluster/kube-util.sh"

prepare-e2e

function calc_next_cidr {
    IFS=/ read -r ip mask <<< "${1}"
    local inc="$((1 << 32-${mask}))"
    local o1 o2 o3 o4
    IFS=. read -r o1 o2 o3 o4 <<< "${ip}"
    local ip_int="$((o1 * 256 ** 3 + o2 * 256 ** 2 + o3 * 256 + o4))"
    local next_ip_int="$((ip_int + inc))"
    local next_ip
    for e in {3..0}
    do
        ((octet = next_ip_int / (256 ** e) ))
        ((next_ip_int -= octet * 256 ** e))
        next_ip+=${delim:-}${octet}
        delim=.
    done
    echo "${next_ip}/${mask}"
}

if [[ "${FEDERATION:-}" == "true" ]];then
    FEDERATION_START_CLUSTER_IP_RANGE="${FEDERATION_START_CLUSTER_IP_RANGE:-10.180.0.0/14}"
    next_cidr=${FEDERATION_START_CLUSTER_IP_RANGE}
    #TODO(colhom): the last cluster that was created in the loop above is the current context.
    # Hence, it will be the cluster that hosts the federated components.
    # In the future, we will want to loop through the all the federated contexts,
    # select each one and call federated-up
    for zone in ${E2E_ZONES}; do
        (
        # This variable should be exported because it is used by scripts which are executed
        # in their own shells.
        export CLUSTER_IP_RANGE=${next_cidr}
        set-federation-zone-vars "$zone"
        test-setup
        )
        # This must be calculated outside the subshell to have an effect in the
        # next iteration of the loop.
        next_cidr="$(calc_next_cidr ${next_cidr})"
    done
    tagfile="${KUBE_ROOT}/federation/manifests/federated-image.tag"
    if [[ ! -f "$tagfile" ]]; then
        echo "FATAL: tagfile ${tagfile} does not exist. Make sure that you have run build/push-federation-images.sh"
        exit 1
    fi
    export FEDERATION_IMAGE_TAG="$(cat "${KUBE_ROOT}/federation/manifests/federated-image.tag")"

    "${KUBE_ROOT}/federation/cluster/federation-up.sh"
else
    test-setup
fi
