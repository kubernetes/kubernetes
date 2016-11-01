#!/bin/bash

# Copyright 2015 The Kubernetes Authors.
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

if [[ "${FEDERATION:-}" == "true" ]]; then
    cur_ip_octet2=180
    # TODO(colhom): the last cluster that was created in the loop above is the current context.
    # Hence, it will be the cluster that hosts the federated components.
    # In the future, we will want to loop through the all the federated contexts,
    # select each one and call federated-up
    for zone in ${E2E_ZONES};do
        (
        export CLUSTER_IP_RANGE="10.${cur_ip_octet2}.0.0/16"
        set-federation-zone-vars "$zone"
        test-setup
        )
        cur_ip_octet2="$((cur_ip_octet2 + 1))"
    done

    # Sets ${CLUSTER_CONTEXT}
    kubeconfig-federation-context "${zone}"

    "${KUBE_ROOT}/federation/cluster/federation-up.sh" "${CLUSTER_CONTEXT}"
else
    test-setup
fi
