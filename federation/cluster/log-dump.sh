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

readonly report_dir="${1:-_artifacts}"
readonly federation_namespace="${2:-federation-system}"
readonly kube_system_namespace="${3:-kube-system}"

readonly apiserver_pod_containers="apiserver etcd"
readonly dns_pod_containers="kubedns dnsmasq sidecar"

echo "Dumping Federation and DNS pod logs to ${report_dir}"

# Get the pods by their labels.
readonly federation_pod_names="$(kubectl get pods -l 'app=federated-cluster' --namespace=${federation_namespace} -o name | tr '\n' ' ')"
readonly dns_pod_names="$(kubectl get pods -l 'k8s-app=kube-dns' --namespace=${kube_system_namespace} -o name | tr '\n' ' ')"

# Dump the Federation pod logs.
for pod_name in ${federation_pod_names}; do
  # The API server pod has two containers
  if [[ "${pod_name}" == *apiserver* ]]; then
    for container in ${apiserver_pod_containers}; do
      kubectl logs "${pod_name}" -c "${container}" --namespace="${federation_namespace}" \
          >"${report_dir}/${pod_name#pod/}-${container}.log"
    done
    continue
  fi

  kubectl logs "${pod_name}" --namespace="${federation_namespace}" \
      >"${report_dir}/${pod_name#pod/}.log"
done

# Dump the DNS pod logs.
for pod_name in ${dns_pod_names}; do
  # As of 3/2017, the only pod that matches the kube-dns label is kube-dns, and
  # it has three containers.
  for container in ${dns_pod_containers}; do
    kubectl logs "${pod_name}" -c "${container}" --namespace="${kube_system_namespace}" \
        >"${report_dir}/${pod_name#pod/}.log"
  done
done
