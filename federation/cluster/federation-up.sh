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
# For $FEDERATION_NAME, $FEDERATION_KUBE_CONTEXT, $HOST_CLUSTER_CONTEXT,
# $KUBEDNS_CONFIGMAP_NAME, $KUBEDNS_CONFIGMAP_NAMESPACE and
# $KUBEDNS_FEDERATION_FLAG.
source "${KUBE_ROOT}/federation/cluster/common.sh"

DNS_ZONE_NAME="${FEDERATION_DNS_ZONE_NAME:-}"
DNS_PROVIDER="${FEDERATION_DNS_PROVIDER:-google-clouddns}"
FEDERATIONS_DOMAIN_MAP="${FEDERATIONS_DOMAIN_MAP:-}"

# get_version returns the version in KUBERNETES_RELEASE or defaults to the
# value in the federation `versions` file.
# TODO(madhusudancs): This is a duplicate of the function in
# federation/develop/develop.sh with a minor difference. This
# function tries to default to the version information in
# _output/federation/versions file where as the one in develop.sh
# tries to default to the version in the kubernetes versions file.
# These functions should be consolidated to read the version from
# kubernetes version defs file.
function get_version() {
  local -r versions_file="${KUBE_ROOT}/_output/federation/versions"

  if [[ -n "${KUBERNETES_RELEASE:-}" ]]; then
    echo "${KUBERNETES_RELEASE//+/_}"
    return
  fi

  if [[ ! -f "${versions_file}" ]]; then
    echo "Couldn't determine the release version: neither the " \
     "KUBERNETES_RELEASE environment variable is set, nor does " \
     "the versions file exist at ${versions_file}"
    exit 1
  fi

  # Read the version back from the versions file if no version is given.
  local -r kube_version="$(cat "${versions_file}" | python -c '\
import json, sys;\
print json.load(sys.stdin)["KUBE_VERSION"]')"

  echo "${kube_version//+/_}"
}

# Initializes the control plane.
# TODO(madhusudancs): Move this to federation/develop.sh.
function init() {
  kube::log::status "Deploying federation control plane for ${FEDERATION_NAME} in cluster ${HOST_CLUSTER_CONTEXT}"

  local -r project="${KUBE_PROJECT:-${PROJECT:-}}"
  local -r kube_registry="${KUBE_REGISTRY:-gcr.io/${project}}"
  local -r kube_version="$(get_version)"

  kube::log::status "DNS_ZONE_NAME: \"${DNS_ZONE_NAME}\", DNS_PROVIDER: \"${DNS_PROVIDER}\""
  kube::log::status "Image: \"${kube_registry}/hyperkube-amd64:${kube_version}\""

  "${KUBE_ROOT}/federation/develop/kubefed.sh" init \
      "${FEDERATION_NAME}" \
      --host-cluster-context="${HOST_CLUSTER_CONTEXT}" \
      --dns-zone-name="${DNS_ZONE_NAME}" \
      --dns-provider="${DNS_PROVIDER}" \
      --image="${kube_registry}/hyperkube-amd64:${kube_version}" \
      --apiserver-arg-overrides="--storage-backend=etcd2" \
      --apiserver-enable-basic-auth=true \
      --apiserver-enable-token-auth=true
}

# join_clusters joins the clusters in the local kubeconfig to federation. The clusters
# and their kubeconfig entries in the local kubeconfig are created while deploying clusters, i.e. when kube-up is run.
function join_clusters() {
  for context in $(federation_cluster_contexts); do
    kube::log::status "Joining cluster with name '${context}' to federation with name '${FEDERATION_NAME}'"

    "${KUBE_ROOT}/federation/develop/kubefed.sh" join \
        "${context}" \
        --host-cluster-context="${HOST_CLUSTER_CONTEXT}" \
        --context="${FEDERATION_NAME}" \
        --secret-name="${context//_/-}"    # Replace "_" by "-"
  done
}

USE_KUBEFED="${USE_KUBEFED:-}"

if [[ "${USE_KUBEFED}" == "true" ]]; then
  init
  join_clusters
else
  export FEDERATION_IMAGE_TAG="$(get_version)"
  create-federation-api-objects
fi
