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

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
# For `kube::log::status` function since it already sources
# "${KUBE_ROOT}/cluster/lib/logging.sh"
source "${KUBE_ROOT}/cluster/common.sh"
# For $FEDERATION_NAME, $FEDERATION_NAMESPACE, $HOST_CLUSTER_CONTEXT,
source "${KUBE_ROOT}/federation/cluster/common.sh"

HYPERKUBE_IMAGE_VERSION="${1}"

host_kubectl="${KUBE_ROOT}/cluster/kubectl.sh --context=${HOST_CLUSTER_CONTEXT} --namespace=${FEDERATION_NAMESPACE}"

function upgrade() {
  local -r project="${KUBE_PROJECT:-${PROJECT:-}}"
  local -r kube_registry="${KUBE_REGISTRY:-gcr.io/${project}}"
  local -r hyperkube_image="${kube_registry}/hyperkube-amd64:${HYPERKUBE_IMAGE_VERSION}"

  kube::log::status "Upgrading federation control plane ${FEDERATION_NAME} with image ${hyperkube_image}"

  # Upgrade apiserver image
  ${host_kubectl} set image deployment/federation-apiserver apiserver=${hyperkube_image}

  # Upgrade controller-manager image
  ${host_kubectl} set image deployment/federation-controller-manager controller-manager=${hyperkube_image}
}

upgrade
