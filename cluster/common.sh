#!/bin/bash

# Copyright 2015 Google Inc. All rights reserved.
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

# Common utilites for kube-up/kube-down

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..

# Generate kubeconfig data for the created cluster.
# Assumed vars:
#   KUBE_USER
#   KUBE_PASSWORD
#   KUBE_MASTER_IP
#   KUBECONFIG
#
#   KUBE_CERT
#   KUBE_KEY
#   CA_CERT
#   CONTEXT
function create-kubeconfig() {
  local kubectl="${KUBE_ROOT}/cluster/kubectl.sh"

  # We expect KUBECONFIG to be defined, which determines the file we write to.
  "${kubectl}" config set-cluster "${CONTEXT}" --server="https://${KUBE_MASTER_IP}" \
                                               --certificate-authority="${CA_CERT}" \
                                               --embed-certs=true
  "${kubectl}" config set-credentials "${CONTEXT}" --username="${KUBE_USER}" \
                                                --password="${KUBE_PASSWORD}" \
                                                --client-certificate="${KUBE_CERT}" \
                                                --client-key="${KUBE_KEY}" \
                                                --embed-certs=true
  "${kubectl}" config set-context "${CONTEXT}" --cluster="${CONTEXT}" --user="${CONTEXT}"
  "${kubectl}" config use-context "${CONTEXT}"  --cluster="${CONTEXT}"

   echo "Wrote config for ${CONTEXT} to ${KUBE_ROOT}/.kubeconfig"
}

# Clear kubeconfig data for a context
# Assumed vars:
#   KUBECONFIG
#   CONTEXT
function clear-kubeconfig() {
  local kubectl="${KUBE_ROOT}/cluster/kubectl.sh"
  "${kubectl}" config unset "clusters.${CONTEXT}"
  "${kubectl}" config unset "users.${CONTEXT}"
  "${kubectl}" config unset "contexts.${CONTEXT}"

  local current
  current=$("${kubectl}" config view -o template --template='{{ index . "current-context" }}')
  if [[ "${current}" == "${CONTEXT}" ]]; then
    "${kubectl}" config unset current-context
  fi

  echo "Cleared config for ${CONTEXT} from ${KUBE_ROOT}/.kubeconfig"
}
