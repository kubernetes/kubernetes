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

# For $FEDERATION_NAME, $FEDERATION_KUBE_CONTEXT, $HOST_CLUSTER_CONTEXT,
# $KUBEDNS_CONFIGMAP_NAME and $KUBEDNS_CONFIGMAP_NAMESPACE.
source "${KUBE_ROOT}/federation/cluster/common.sh"

# unjoin_clusters unjoins all the clusters from federation.
function unjoin_clusters() {
  for context in $(federation_cluster_contexts); do
    kube::log::status "Unjoining cluster \"${context}\" from federation \"${FEDERATION_NAME}\""

    "${KUBE_ROOT}/federation/develop/kubefed.sh" unjoin \
        "${context}" \
        --context="${FEDERATION_KUBE_CONTEXT}" \
        --host-cluster-context="${HOST_CLUSTER_CONTEXT}"

    # Delete kube-dns configmap that contains federation
    # configuration from each cluster.
    # TODO: This shouldn't be required after
    # https://github.com/kubernetes/kubernetes/pull/39338.
    # Remove this after the PR is merged.
    kube::log::status "Deleting \"kube-dns\" ConfigMap from \"kube-system\" namespace in cluster \"${context}\""
    "${KUBE_ROOT}/cluster/kubectl.sh" delete configmap \
        --context="${context}" \
        --namespace="${KUBEDNS_CONFIGMAP_NAMESPACE}" \
        "${KUBEDNS_CONFIGMAP_NAME}"
  done
}

unjoin_clusters

if cleanup-federation-api-objects; then
  # TODO(madhusudancs): This is an arbitrary amount of sleep to give
  # Kubernetes clusters enough time to delete the underlying cloud
  # provider resources corresponding to the Kubernetes resources we
  # deleted as part of the test tear downs. It is shameful that we
  # are doing this, but this is just a bandage to stop the bleeding.
  # Please don't use this pattern anywhere. Remove this when proper
  # cloud provider cleanups are implemented in the individual test
  # `AfterEach` blocks.
  # Also, we wait only if the cleanup succeeds.
  sleep 2m
else
  echo "Couldn't cleanup federation api objects"
fi
