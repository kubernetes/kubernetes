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

# federation_clusters returns a list of all the clusters in
# federation, if at all the federation control plane exists
# and there are any clusters registerd.
function federation_clusters() {
  if clusters=$("${KUBE_ROOT}/cluster/kubectl.sh" \
      --context="${FEDERATION_KUBE_CONTEXT}" \
      -o jsonpath --template '{.items[*].metadata.name}' \
      get clusters) ; then
    echo ${clusters}
    return
  fi
  echo ""
}

# unjoin_clusters unjoins all the clusters from federation.
function unjoin_clusters() {
  # Unjoin only those clusters that are registered with the
  # given federation. This is slightly different than
  # joining clusters where we join all the clusters in the
  # current kubeconfig with the "federation" prefix.
  for context in $(federation_clusters); do
    kube::log::status "Unjoining cluster \"${context}\" from federation \"${FEDERATION_NAME}\""

    "${KUBE_ROOT}/federation/develop/kubefed.sh" unjoin \
        "${context}" \
        --context="${FEDERATION_KUBE_CONTEXT}" \
        --host-cluster-context="${HOST_CLUSTER_CONTEXT}"
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
  kube::log::status "Waiting for 2 minutes to allow controllers to clean up federation components..."
  sleep 2m
else
  echo "Couldn't cleanup federation api objects"
fi
