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

# A library of helper functions for a pre-existing Kubernetes cluster
KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
source "${KUBE_ROOT}/cluster/common.sh"
source "${KUBE_ROOT}/cluster/lib/util.sh"

function detect-project() {
    if [[ -z "${MASTER_IP-}" ]]; then
	echo "Set 'MASTER_IP' to the instance assigned to be the Kubernetes master" 1>&2
	exit 1
    fi

    if [[ -z "${PROJECT-}" ]]; then
	PROJECT="kubemark"
    fi

    if [[ -z "${SERVICE_CLUSTER_IP_RANGE-}" ]]; then
	octets=($(echo "${MASTER_IP}" | sed -e 's|/.*||' -e 's/\./ /g'))
	octets[3]=0
	cluster_range=$(echo "${octets[*]}" | sed 's/ /./g')
	SERVICE_CLUSTER_IP_RANGE="${SERVICE_CLUSTER_IP_RANGE:-$cluster_range}"
    fi
}

function run-cmd-with-retries {
  RETRIES="${RETRIES:-3}"
  for attempt in $(seq 1 ${RETRIES}); do
    local -r result=$("$@" 2>&1)
    local -r ret_val="$?"
    echo "${result}"
    if [[ "${ret_val}" -ne "0" ]]; then
      echo -e "${color_yellow}Attempt $attempt failed to $1 $2 $3. Retrying.${color_norm}" >& 2
      sleep $(($attempt * 5))
    else
      echo -e "${color_green}Succeeded to $1 $2 $3.${color_norm}"
      return 0
    fi
  done
  echo -e "${color_red}Failed to $1 $2 $3.${color_norm}" >& 2
  exit 1
}

function execute-cmd-on-master-with-retries() {
  run-cmd-with-retries ssh kubernetes@"${MASTER_IP}" $@
}

function create-certs {
  execute-cmd-on-master-with-retries "sudo cat /etc/kubernetes/admin.conf" > /tmp/kubeconfig
  CA_CERT_BASE64=$(cat /tmp/kubeconfig | grep certificate-authority | awk '{print $2}')
  KUBELET_CERT_BASE64=$(cat /tmp/kubeconfig | grep client-certificate-data | awk '{print $2}')
  KUBELET_KEY_BASE64=$(cat /tmp/kubeconfig | grep client-key-data | awk '{print $2}')

  # Local kubeconfig.kubemark vars
  KUBECFG_CERT_BASE64="${KUBELET_CERT_BASE64}"
  KUBECFG_KEY_BASE64="${KUBELET_KEY_BASE64}"

  # The pre-existing Kubernetes cluster already has these setup
  # Set these vars but don't use them
  CA_KEY_BASE64=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
  MASTER_CERT_BASE64=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
  MASTER_KEY_BASE64=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
  KUBEAPISERVER_CERT_BASE64=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
  KUBEAPISERVER_KEY_BASE64=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
}
