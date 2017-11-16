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

# A library of helper functions for landing kubemark containers on a
# pre-existing Kubernetes master. See test/kubemark/pre-existing/README.md
# for me details on using a pre-existing provider.

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..

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
  rm /tmp/kubeconfig

  execute-cmd-on-pre-existing-master-with-retries "sudo cat /etc/kubernetes/admin.conf" > /tmp/kubeconfig
  CA_CERT_BASE64=$(cat /tmp/kubeconfig | grep certificate-authority | awk '{print $2}' | head -n 1)
  KUBELET_CERT_BASE64=$(cat /tmp/kubeconfig | grep client-certificate-data | awk '{print $2}' | head -n 1)
  KUBELET_KEY_BASE64=$(cat /tmp/kubeconfig | grep client-key-data | awk '{print $2}' | head -n 1)

  # Local kubeconfig.kubemark vars
  KUBECFG_CERT_BASE64="${KUBELET_CERT_BASE64}"
  KUBECFG_KEY_BASE64="${KUBELET_KEY_BASE64}"

  # The pre-existing Kubernetes master already has these setup
  # Set these vars but don't use them
  CA_KEY_BASE64=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
  MASTER_CERT_BASE64=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
  MASTER_KEY_BASE64=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
  KUBEAPISERVER_CERT_BASE64=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
  KUBEAPISERVER_KEY_BASE64=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
}
