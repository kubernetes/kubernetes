#!/usr/bin/env bash

# Copyright 2020 The Kubernetes Authors.
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

# This script contains the helper functions that each provider hosting
# Kubermark must implement to use test/kubemark/start-kubemark.sh and
# test/

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/../../..

source "${KUBE_ROOT}/test/kubemark/common/util.sh"

# This function should authenticate docker to be able to read/write to
# the right container registry (needed for pushing kubemark image).
function authenticate-docker {
    echo "AKS Kubemark provider: no-op docker login" 1>&2
}

# This function should create kubemark master and write kubeconfig to
# "${RESOURCE_DIRECTORY}/kubeconfig.kubemark".
# If a cluster uses private master IP, create-kubemark-master might also write
# a second kubeconfig to "${RESOURCE_DIRECTORY}/kubeconfig-internal.kubemark".
# The difference between these two kubeconfigs is that the internal one uses
# private master IP, which might be better suited for setting up hollow nodes.
function create-kubemark-master {
    set -x
    echo "Creating cluster..."
    if [ "$USE_EXISTING" != "true" ]; then
      GROUP_EXISTS=$(az group list -o tsv --query "[?name=='${KUBEMARK_RESOURCE_GROUP}'].name")
      if [ -z "$GROUP_EXISTS" ]; then 
        az group create -g "${KUBEMARK_RESOURCE_GROUP}" --location "${KUBEMARK_LOCATION}"
      fi
      az aks create \
          -g "${KUBEMARK_RESOURCE_GROUP}" \
          -n "${KUBEMARK_RESOURCE_NAME}" \
          --load-balancer-sku Standard \
          --kubernetes-version "${KUBEMARK_KUBE_VERSION}" \
          --location "${KUBEMARK_LOCATION}" \
          --node-osdisk-size "${KUBEMARK_OS_DISK}" \
          --node-vm-size "${KUBEMARK_NODE_SKU}" \
          --node-count "${KUBEMARK_REAL_NODES}"
    fi;
    az aks get-credentials \
        -g "${KUBEMARK_RESOURCE_GROUP}" \
        -n "${KUBEMARK_RESOURCE_NAME}" \
        -f "${RESOURCE_DIRECTORY}/kubeconfig.kubemark"
    FQDN=$(az aks show \
      -g "${KUBEMARK_RESOURCE_GROUP}" \
      -n "${KUBEMARK_RESOURCE_NAME}" \
      --query 'fqdn' -o tsv)
    export MASTER_IP="$FQDN"
    export MASTER_INTERNAL_IP="$MASTER_IP"
    export KUBECONFIG="${RESOURCE_DIRECTORY}/kubeconfig.kubemark"
}

# This function should delete kubemark master.
function delete-kubemark-master {
  set -x
  echo "Deleting cluster..."
  CLUSTER_EXISTS="$(az resource list -o tsv --query "[?name=='${KUBEMARK_RESOURCE_NAME}' && resourceGroup=='${KUBEMARK_RESOURCE_GROUP}'].id")"

  if [ "$USE_EXISTING" != "true" ]; then
    if [ -n "$CLUSTER_EXISTS" ]; then
      az aks delete -g "${KUBEMARK_RESOURCE_GROUP}" -n "${KUBEMARK_RESOURCE_NAME}" -y
    fi
  fi
}

# This function should return node labels.
function calculate-node-labels {
  echo ""
}

# Common colors used throughout the kubemark scripts
if [[ -z "${color_start-}" ]]; then
  declare -r color_start="\033["
  # shellcheck disable=SC2034
  declare -r color_red="${color_start}0;31m"
  # shellcheck disable=SC2034
  declare -r color_yellow="${color_start}0;33m"
  # shellcheck disable=SC2034
  declare -r color_green="${color_start}0;32m"
  # shellcheck disable=SC2034
  declare -r color_blue="${color_start}1;34m"
  # shellcheck disable=SC2034
  declare -r color_cyan="${color_start}1;36m"
  # shellcheck disable=SC2034
  declare -r color_norm="${color_start}0m"
fi
