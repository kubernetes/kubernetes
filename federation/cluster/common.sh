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

# required:
# KUBE_ROOT: path of the root of the Kubernetes repository

: "${KUBE_ROOT?Must set KUBE_ROOT env var}"

# Provides the $KUBERNETES_PROVIDER, kubeconfig-federation-context()
# and detect-project function
source "${KUBE_ROOT}/cluster/kube-util.sh"

# kubefed configuration
FEDERATION_NAME="${FEDERATION_NAME:-e2e-federation}"
FEDERATION_NAMESPACE=${FEDERATION_NAMESPACE:-federation-system}
FEDERATION_KUBE_CONTEXT="${FEDERATION_KUBE_CONTEXT:-${FEDERATION_NAME}}"
HOST_CLUSTER_ZONE="${FEDERATION_HOST_CLUSTER_ZONE:-}"
# If $HOST_CLUSTER_ZONE isn't specified, arbitrarily choose
# last zone as the host cluster zone.
if [[ -z "${HOST_CLUSTER_ZONE}" ]]; then
  E2E_ZONES_ARR=(${E2E_ZONES:-})
  if [[ ${#E2E_ZONES_ARR[@]} > 0 ]]; then
    HOST_CLUSTER_ZONE=${E2E_ZONES_ARR[-1]}
  fi
fi

HOST_CLUSTER_CONTEXT="${FEDERATION_HOST_CLUSTER_CONTEXT:-}"
if [[ -z "${HOST_CLUSTER_CONTEXT}" ]]; then
  # Sets ${CLUSTER_CONTEXT}
  if [[ -z "${HOST_CLUSTER_ZONE:-}" ]]; then
    echo "At least one of FEDERATION_HOST_CLUSTER_CONTEXT, FEDERATION_HOST_CLUSTER_ZONE or E2E_ZONES is required."
    exit 1
  fi
  kubeconfig-federation-context "${HOST_CLUSTER_ZONE:-}"
  HOST_CLUSTER_CONTEXT="${CLUSTER_CONTEXT}"
fi

function federation_cluster_contexts() {
  local -r contexts=$("${KUBE_ROOT}/cluster/kubectl.sh" config get-contexts -o name)
  federation_contexts=()
  for context in ${contexts}; do
    # Skip federation context
    if [[ "${context}" == "${FEDERATION_NAME}" ]]; then
      continue
    fi
    # Skip contexts not beginning with "federation"
    if [[ "${context}" != federation* ]]; then
      continue
    fi
    federation_contexts+=("${context}")
  done
  echo ${federation_contexts[@]}
}


source "${KUBE_ROOT}/cluster/common.sh"

host_kubectl="${KUBE_ROOT}/cluster/kubectl.sh --namespace=${FEDERATION_NAMESPACE}"

function cleanup-federation-api-objects {
  # This is a cleanup function. We cannot stop on errors here. So disable
  # errexit in this function.
  set +o errexit

  echo "Cleaning Federation control plane objects"
  kube::log::status "Removing namespace \"${FEDERATION_NAMESPACE}\" from \"${FEDERATION_KUBE_CONTEXT}\""
  # Try deleting until the namespace is completely gone.
  while $host_kubectl --context="${FEDERATION_KUBE_CONTEXT}" delete namespace "${FEDERATION_NAMESPACE}" >/dev/null 2>&1; do
    # It is usually slower to remove a namespace because it involves
    # performing a cascading deletion of all the resources in the
    # namespace. So we sleep a little longer than other resources
    # before retrying
    sleep 5
  done
  kube::log::status "Removed namespace \"${FEDERATION_NAMESPACE}\" from \"${FEDERATION_KUBE_CONTEXT}\""

  # This is a big hammer. We get rid of federation-system namespace from
  # all the clusters
  for context in $(federation_cluster_contexts); do
    (
      local -r role="federation-controller-manager:${FEDERATION_NAME}-${context}-${HOST_CLUSTER_CONTEXT}"
      kube::log::status "Removing namespace \"${FEDERATION_NAMESPACE}\", cluster role \"${role}\" and cluster role binding \"${role}\" from \"${context}\""
      # Try deleting until the namespace is completely gone.
      while $host_kubectl --context="${context}" delete namespace "${FEDERATION_NAMESPACE}" >/dev/null 2>&1; do
        # It is usually slower to remove a namespace because it involves
        # performing a cascading deletion of all the resources in the
        # namespace. So we sleep a little longer than other resources
        # before retrying
        sleep 5
      done
      kube::log::status "Removed namespace \"${FEDERATION_NAMESPACE}\" from \"${context}\""

      while $host_kubectl --context="${context}" delete clusterrole "${role}" >/dev/null 2>&1; do
        sleep 2
      done
      kube::log::status "Removed cluster role \"${role}\" from \"${context}\""

      while $host_kubectl --context="${context}" delete clusterrolebinding "${role}" >/dev/null 2>&1; do
        sleep 2
      done
      kube::log::status "Removed cluster role binding \"${role}\" from \"${context}\""
    ) &
  done
  wait
  set -o errexit
}
