#!/bin/bash

# Copyright 2014 The Kubernetes Authors.
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

KUBE_ROOT=$(readlink -m $(dirname "${BASH_SOURCE}")/../../)

# For $FEDERATION_KUBE_CONTEXT, $HOST_CLUSTER_CONTEXT,
# $KUBEDNS_CONFIGMAP_NAME and $KUBEDNS_CONFIGMAP_NAMESPACE.
source "${KUBE_ROOT}/federation/cluster/common.sh"

# join_cluster_to_federation joins the clusters in the local kubeconfig to federation. The clusters
# and their kubeconfig entries in the local kubeconfig are created while deploying clusters, i.e. when kube-up is run.
function unjoin_clusters() {
  "${host_kubectl}" config use-context "${FEDERATION_KUBE_CONTEXT}"

  local -r clusters=$("${host_kubectl}" -o jsonpath --template '{.items[*].metadata.name}')
  for cluster in ${clusters}; do
    kube::log::status "Unjoining cluster \"${cluster}\" from federation \"${FEDERATION_NAME}\""

    "${KUBE_ROOT}/federation/develop/kubefed.sh" unjoin \
        "${cluster}" \
        --host-cluster-context="${HOST_CLUSTER_CONTEXT}"

    # Create kube-dns configmap in each cluster for kube-dns to accept
    # federation queries.
    # TODO: This shouldn't be required after
    # https://github.com/kubernetes/kubernetes/pull/39338.
    # Remove this after the PR is merged.
    "${host_kubectl}" delete configmap \
        --namespace="${KUBEDNS_CONFIGMAP_NAMESPACE}" \
        "${KUBEDNS_CONFIGMAP_NAME}"
  done
}

cleanup-federation-api-objects

"${host_kubectl}" delete namespace "${FEDERATION_NAMESPACE}"

