#!/bin/bash

# Copyright 2017 The Kubernetes Authors.
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

# Call this to dump all Federation pod logs into the folder specified in $1
# (defaults to _artifacts).

set -o errexit
set -o nounset
set -o pipefail

# For FEDERATION_NAMESPACE
KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
source "${KUBE_ROOT}/federation/cluster/common.sh"

readonly REPORT_DIR="${1:-_artifacts}"
OUTPUT_DIR="${REPORT_DIR}/federation"

# Dumps logs for all pods in a federation.
function dump_federation_pod_logs() {
  local -r federation_pod_names_string="$(kubectl get pods -l 'app=federated-cluster' --namespace=${FEDERATION_NAMESPACE} -o name)"
  if [[ -z "${federation_pod_names_string}" ]]; then
    return
  fi

  local -r federation_pod_names=(${federation_pod_names_string})
  for pod_name in ${federation_pod_names[@]}; do
    # The API server pod has two containers
    if [[ "${pod_name}" == *apiserver* ]]; then
      dump_apiserver_pod_logs "${pod_name}"
      continue
    fi

    kubectl logs "${pod_name}" --namespace="${FEDERATION_NAMESPACE}" \
        >"${OUTPUT_DIR}/${pod_name#pods/}.log"
  done
}

# Dumps logs from all containers in an API server pod.
# Arguments:
# - the name of the API server pod, with a pods/ prefix.
function dump_apiserver_pod_logs() {
  local -r apiserver_pod_containers=(apiserver etcd)
  for container in ${apiserver_pod_containers[@]}; do
    kubectl logs "${1}" -c "${container}" --namespace="${FEDERATION_NAMESPACE}" \
        >"${OUTPUT_DIR}/${1#pods/}-${container}.log"
  done
}

# Dumps logs from all containers in the DNS pods.
# TODO: This currently only grabs DNS pod logs from the host cluster. It should
# grab those logs from all clusters in the federation.
function dump_dns_pod_logs() {
  local -r dns_pod_names_string="$(kubectl get pods -l 'k8s-app=kube-dns' --namespace=kube-system -o name)"
  if [[ -z "${dns_pod_names_string}" ]]; then
    return
  fi

  local -r dns_pod_names=(${dns_pod_names_string})
  local -r dns_pod_containers=(kubedns dnsmasq sidecar)

  for pod_name in ${dns_pod_names[@]}; do
    # As of 3/2017, the only pod that matches the kube-dns label is kube-dns, and
    # it has three containers.
    for container in ${dns_pod_containers[@]}; do
      kubectl logs "${pod_name}" -c "${container}" --namespace=kube-system \
          >"${OUTPUT_DIR}/${pod_name#pods/}-${container}.log"
    done
  done
}


echo "Dumping Federation and DNS pod logs to ${REPORT_DIR}"
mkdir -p "${OUTPUT_DIR}"

dump_federation_pod_logs
dump_dns_pod_logs
