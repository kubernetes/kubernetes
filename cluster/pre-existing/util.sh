#!/usr/bin/env bash

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

# A library of helper functions for landing kubemark containers on a
# pre-existing Kubernetes master. See test/kubemark/pre-existing/README.md
# for me details on using a pre-existing provider.

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/../..

source "${KUBE_ROOT}/cluster/common.sh"
source "${KUBE_ROOT}/hack/lib/util.sh"

function detect-project() {
  if [[ -z "${MASTER_IP:-}" ]]; then
    echo "Set 'MASTER_IP' to the instance assigned to be the Kubernetes master" 1>&2
    exit 1
  fi

  if [[ -z "${PROJECT:-}" ]]; then
    echo "Set 'PROJECT' to the name of the container project: $CONTAINER_REGISTRY/$PROJECT/kubemark" >&2
    exit 1
  fi

  if [[ -z "${SERVICE_CLUSTER_IP_RANGE:-}" ]]; then
    cluster_range=$(echo "${MASTER_IP}" | awk -F '.' '{printf("%d.%d.%d.0", $1, $2, $3)}')
    SERVICE_CLUSTER_IP_RANGE="${SERVICE_CLUSTER_IP_RANGE:-$cluster_range/16}"
  fi
}

function create-certs {
  execute-cmd-on-pre-existing-master-with-retries 'sudo cat /etc/kubernetes/admin.conf' > /tmp/kubeconfig

  # CA_CERT_BASE64, KUBELET_CERT_BASE64 and KUBELET_KEY_BASE64 might be used
  # in test/kubemark/iks/util.sh
  # If it becomes clear that the variables are not used anywhere, then we can
  # remove them:
  CA_CERT_BASE64=$(grep certificate-authority /tmp/kubeconfig | awk '{print $2}' | head -n 1)
  KUBELET_CERT_BASE64=$(grep client-certificate-data /tmp/kubeconfig | awk '{print $2}' | head -n 1)
  KUBELET_KEY_BASE64=$(grep client-key-data /tmp/kubeconfig | awk '{print $2}' | head -n 1)
  export CA_CERT_BASE64
  export KUBELET_CERT_BASE64
  export KUBELET_KEY_BASE64
}
