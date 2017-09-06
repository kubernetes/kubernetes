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

# This script will source the default skeleton helper functions, then sources
# cluster/${KUBERNETES_PROVIDER}/util.sh where KUBERNETES_PROVIDER, if unset,
# will use its default value (gce).

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..

source "${KUBE_ROOT}/cluster/skeleton/util.sh"

if [[ -n "${KUBERNETES_CONFORMANCE_TEST:-}" ]]; then
    KUBERNETES_PROVIDER=""
else
    KUBERNETES_PROVIDER="${KUBERNETES_PROVIDER:-gce}"
fi

# PROVIDER_VARS is a list of cloud provider specific variables. Note:
# this is a list of the _names_ of the variables, not the value of the
# variables. Providers can add variables to be appended to kube-env.
# (see `build-kube-env`).
PROVIDER_VARS=""

PROVIDER_UTILS="${KUBE_ROOT}/cluster/${KUBERNETES_PROVIDER}/util.sh"
if [ -f ${PROVIDER_UTILS} ]; then
    source "${PROVIDER_UTILS}"
fi

# Federation utils

# Sets the kubeconfig context value for the current cluster.
# Args:
#   $1: zone (required)
#
# Vars set:
#   CLUSTER_CONTEXT
function kubeconfig-federation-context() {
  if [[ -z "${1:-}" ]]; then
    echo "zone parameter is required"
    exit 1
  fi
  CLUSTER_CONTEXT="federation-e2e-${KUBERNETES_PROVIDER}-${1}"
}

# Should NOT be called within the global scope, unless setting the desired global zone vars
# This function is currently NOT USED in the global scope
function set-federation-zone-vars {
  zone="$1"
  kubeconfig-federation-context "${zone}"
  export OVERRIDE_CONTEXT="${CLUSTER_CONTEXT}"
  echo "Setting zone vars to: $OVERRIDE_CONTEXT"
  if [[ "$KUBERNETES_PROVIDER" == "gce"  ]];then
    # This needs a revamp, but for now e2e zone name is used as the unique
    # cluster identifier in our e2e tests and we will continue to use that
    # pattern.
    export CLUSTER_NAME="${zone}"

    export KUBE_GCE_ZONE="${zone}"
    # gcloud has a 61 character limit, and for firewall rules this
    # prefix gets appended to itself, with some extra information
    # need tot keep it short
    export KUBE_GCE_INSTANCE_PREFIX="${USER}-${zone}"

  elif [[ "$KUBERNETES_PROVIDER" == "gke"  ]];then

    export CLUSTER_NAME="${USER}-${zone}"

  elif [[ "$KUBERNETES_PROVIDER" == "aws"  ]];then

    export KUBE_AWS_ZONE="$zone"
    export KUBE_AWS_INSTANCE_PREFIX="${USER}-${zone}"

    # WARNING: This is hack
    # After KUBE_AWS_INSTANCE_PREFIX is changed,
    # we need to make sure the config-xxx.sh file is
    # re-sourced so the change propogates to dependent computed values
    # (eg: MASTER_SG_NAME, NODE_SG_NAME, etc)
    source "${KUBE_ROOT}/cluster/aws/util.sh"
  else
    echo "Provider \"${KUBERNETES_PROVIDER}\" is not supported"
    exit 1
  fi
}
