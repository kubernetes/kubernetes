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

# This script is only used for e2e tests! Don't use it in production!
# This is also a temporary bridge to slowly switch over everything to
# federation/develop.sh. Carefully moving things step-by-step, ensuring
# things don't break.
# TODO(madhusudancs): Remove this script and its dependencies.


KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
# For `kube::log::status` function since it already sources
# "${KUBE_ROOT}/cluster/lib/logging.sh" and DEFAULT_KUBECONFIG
source "${KUBE_ROOT}/cluster/common.sh"
# For $KUBE_PLATFORM, $KUBE_ARCH, $KUBE_BUILD_STAGE,
# $FEDERATION_PUSH_REPO_BASE and $FEDERATION_NAMESPACE.
source "${KUBE_ROOT}/federation/cluster/common.sh"

FEDERATION_NAME="${FEDERATION_NAME:-e2e-federation}"
FEDERATION_KUBE_CONTEXT="${FEDERATION_KUBE_CONTEXT:-e2e-federation}"
DNS_ZONE_NAME="${FEDERATION_DNS_ZONE_NAME:-}"
HOST_CLUSTER_CONTEXT="${FEDERATION_HOST_CLUSTER_CONTEXT:-${1}}"
readonly CLIENT_BIN_DIR="${KUBE_ROOT}/_output/${KUBE_BUILD_STAGE}/client/${KUBE_PLATFORM}-${KUBE_ARCH}/kubernetes/client/bin"
kubefed="${CLIENT_BIN_DIR}/kubefed"
kubectl="${CLIENT_BIN_DIR}/kubectl"

# Initializes the control plane.
# TODO(madhusudancs): Move this to federation/develop.sh.
function init() {
  kube::log::status "Deploying federation control plane for ${FEDERATION_NAME} in cluster ${HOST_CLUSTER_CONTEXT}"

  local -r project="${KUBE_PROJECT:-${PROJECT:-}}"
  local -r kube_registry="${KUBE_REGISTRY:-gcr.io/${project}}"
  local -r kube_version="${KUBERNETES_RELEASE:-}"

  ${kubefed} init \
      "${FEDERATION_NAME}" \
      --host-cluster-context="${HOST_CLUSTER_CONTEXT}" \
      --dns-zone-name="${DNS_ZONE_NAME}" \
      --image="${kube_registry}/hyperkube-amd64:${kube_version}"
}

# create_cluster_secrets creates the secrets containing the kubeconfigs
# of the participating clusters in the host cluster. The kubeconfigs itself
# are created while deploying clusters, i.e. when kube-up is run.
function create_cluster_secrets() {
  local -r kubeconfig_dir="$(dirname ${DEFAULT_KUBECONFIG})"
  local -r base_dir="${kubeconfig_dir}/federation/kubernetes-apiserver"

  # Create secrets with all the kubernetes-apiserver's kubeconfigs.
  for dir in $(ls "${base_dir}"); do
    # We create a secret with the same name as the directory name (which is
    # same as cluster name in kubeconfig).
    # Massage the name so that it is valid (should not contain "_" and max 253
    # chars)
    name=$(echo "${dir}" | sed -e "s/_/-/g")  # Replace "_" by "-"
    name=${name:0:252}
    kube::log::status "Creating secret with name: ${name} in namespace ${FEDERATION_NAMESPACE}"
    ${kubectl} create secret generic ${name} --from-file="${base_dir}/${dir}/kubeconfig" --namespace="${FEDERATION_NAMESPACE}"
  done
}

USE_KUBEFED="${USE_KUBEFED:-}"
if [[ "${USE_KUBEFED}" == "true" ]]; then
  init
  create_cluster_secrets
else
  export FEDERATION_IMAGE_TAG="$(echo ${KUBERNETES_RELEASE:-} | tr + _)"
  create-federation-api-objects
fi
