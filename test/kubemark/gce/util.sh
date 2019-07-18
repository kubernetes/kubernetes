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

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/../../..

source "${KUBE_ROOT}/test/kubemark/common/util.sh"

# Wrapper for gcloud compute, running it $RETRIES times in case of failures.
# Args:
# $@: all stuff that goes after 'gcloud compute'
function run-gcloud-compute-with-retries {
  run-cmd-with-retries gcloud compute "$@"
}

function authenticate-docker {
  echo "Configuring registry authentication"
  mkdir -p "${HOME}/.docker"
  gcloud beta auth configure-docker -q
}

function create-kubemark-master {
  # We intentionally override env vars in subshell to preserve original values.
  # shellcheck disable=SC2030,SC2031
  (
    # All calls to e2e-grow-cluster must share temp dir with initial e2e-up.sh.
    kube::util::ensure-temp-dir
    export KUBE_TEMP="${KUBE_TEMP}"

    export KUBECONFIG="${RESOURCE_DIRECTORY}/kubeconfig.kubemark"
    export CLUSTER_NAME="${CLUSTER_NAME}-kubemark"
    export KUBE_CREATE_NODES=false
    export KUBE_GCE_INSTANCE_PREFIX="${KUBE_GCE_INSTANCE_PREFIX}-kubemark"

    # Even if the "real cluster" is private, we shouldn't manage cloud nat.
    export KUBE_GCE_PRIVATE_CLUSTER=false

    # Quite tricky cidr setup: we set KUBE_GCE_ENABLE_IP_ALIASES=true to avoid creating
    # cloud routes and RangeAllocator to assign cidrs by kube-controller-manager.
    export KUBE_GCE_ENABLE_IP_ALIASES=true
    export KUBE_GCE_NODE_IPAM_MODE=RangeAllocator

    # Disable all addons. They are running outside of the kubemark cluster.
    export KUBE_ENABLE_CLUSTER_AUTOSCALER=false
    export KUBE_ENABLE_CLUSTER_DNS=false
    export KUBE_ENABLE_NODE_LOGGING=false
    export KUBE_ENABLE_METRICS_SERVER=false
    export KUBE_ENABLE_CLUSTER_MONITORING="none"
    export KUBE_ENABLE_L7_LOADBALANCING="none"

    # Unset env variables set by kubetest for 'root cluster'. We need recompute them
    # for kubemark master.
    # TODO(mborsz): Figure out some better way to filter out such env variables than
    # listing them here.
    unset MASTER_SIZE MASTER_DISK_SIZE MASTER_ROOT_DISK_SIZE

    # Set kubemark-specific overrides:
    # for each defined env KUBEMARK_X=Y call export X=Y.
    for var in ${!KUBEMARK_*}; do
      dst_var=${var#KUBEMARK_}
      val=${!var}
      echo "Setting ${dst_var} to '${val}'"
      export "${dst_var}"="${val}"
    done

    "${KUBE_ROOT}/hack/e2e-internal/e2e-up.sh"

    if [[ "${KUBEMARK_HA_MASTER:-}" == "true" && -n "${KUBEMARK_MASTER_ADDITIONAL_ZONES:-}" ]]; then
        for KUBE_GCE_ZONE in ${KUBEMARK_MASTER_ADDITIONAL_ZONES}; do
          KUBE_GCE_ZONE="${KUBE_GCE_ZONE}" KUBE_REPLICATE_EXISTING_MASTER=true \
            "${KUBE_ROOT}/hack/e2e-internal/e2e-grow-cluster.sh"
        done
    fi
    )
}

function delete-kubemark-master {
  # We intentionally override env vars in subshell to preserve original values.
  # shellcheck disable=SC2030,SC2031
  (
    export CLUSTER_NAME="${CLUSTER_NAME}-kubemark"
    export KUBE_GCE_INSTANCE_PREFIX="${KUBE_GCE_INSTANCE_PREFIX}-kubemark"

    export KUBE_DELETE_NETWORK=false
    # Even if the "real cluster" is private, we shouldn't manage cloud nat.
    export KUBE_GCE_PRIVATE_CLUSTER=false

    if [[ "${KUBEMARK_HA_MASTER:-}" == "true" && -n "${KUBEMARK_MASTER_ADDITIONAL_ZONES:-}" ]]; then
      for KUBE_GCE_ZONE in ${KUBEMARK_MASTER_ADDITIONAL_ZONES}; do
        KUBE_GCE_ZONE="${KUBE_GCE_ZONE}" KUBE_REPLICATE_EXISTING_MASTER=true \
          "${KUBE_ROOT}/hack/e2e-internal/e2e-shrink-cluster.sh"
      done
    fi

    "${KUBE_ROOT}/hack/e2e-internal/e2e-down.sh"
  )
}
