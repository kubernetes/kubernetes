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
    export KUBECONFIG_INTERNAL="${RESOURCE_DIRECTORY}/kubeconfig-internal.kubemark"
    export CLUSTER_NAME="${CLUSTER_NAME}-kubemark"
    export KUBE_CREATE_NODES=false
    export KUBE_GCE_INSTANCE_PREFIX="${KUBE_GCE_INSTANCE_PREFIX}-kubemark"

    # Quite tricky cidr setup: we set KUBE_GCE_ENABLE_IP_ALIASES=true to avoid creating
    # cloud routes and RangeAllocator to assign cidrs by kube-controller-manager.
    export KUBE_GCE_ENABLE_IP_ALIASES=true
    export KUBE_GCE_NODE_IPAM_MODE=RangeAllocator

    # Disable all addons. They are running outside of the kubemark cluster.
    export KUBE_ENABLE_CLUSTER_AUTOSCALER=false
    export KUBE_ENABLE_CLUSTER_DNS=false
    export KUBE_ENABLE_NODE_LOGGING=false
    export KUBE_ENABLE_METRICS_SERVER=false
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

    # The e2e-up.sh script is not sourced, so we don't have access to variables that
    # it sets. Instead, we read data which was written to the KUBE_TEMP directory.
    # The cluster-location is either ZONE (say us-east1-a) or REGION (say us-east1).
    # To get REGION from location, only first two parts are matched.
    REGION=$(grep -o "^[a-z]*-[a-z0-9]*" "${KUBE_TEMP}"/cluster-location.txt)
    MASTER_NAME="${KUBE_GCE_INSTANCE_PREFIX}"-master

    if [[ ${GCE_PRIVATE_CLUSTER:-} == "true" ]]; then
      MASTER_INTERNAL_IP=$(gcloud compute addresses describe "${MASTER_NAME}-internal-ip" \
          --project "${PROJECT}" --region "${REGION}" -q --format='value(address)')
    fi
    MASTER_IP=$(gcloud compute addresses describe "${MASTER_NAME}-ip" \
        --project "${PROJECT}" --region "${REGION}" -q --format='value(address)')

    # If cluster uses private master IP, two kubeconfigs are created:
    # - kubeconfig with public IP, which will be used to connect to the cluster
    #     from outside of the cluster network
    # - kubeconfig with private IP (called internal kubeconfig), which will be
    #      used to create hollow nodes.
    #
    # Note that hollow nodes might use either of these kubeconfigs, but
    # using internal one is better from performance and cost perspective, since
    # traffic does not need to go through Cloud NAT.
    if [[ -n "${MASTER_INTERNAL_IP:-}" ]]; then
      echo "Writing internal kubeconfig to '${KUBECONFIG_INTERNAL}'"
      ip_regexp=${MASTER_IP//./\\.} # escape ".", so that sed won't treat it as "any char"
      sed "s/${ip_regexp}/${MASTER_INTERNAL_IP}/g" "${KUBECONFIG}" > "${KUBECONFIG_INTERNAL}"
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

    if [[ "${KUBEMARK_HA_MASTER:-}" == "true" && -n "${KUBEMARK_MASTER_ADDITIONAL_ZONES:-}" ]]; then
      for KUBE_GCE_ZONE in ${KUBEMARK_MASTER_ADDITIONAL_ZONES}; do
        KUBE_GCE_ZONE="${KUBE_GCE_ZONE}" KUBE_REPLICATE_EXISTING_MASTER=true \
          "${KUBE_ROOT}/hack/e2e-internal/e2e-shrink-cluster.sh"
      done
    fi

    "${KUBE_ROOT}/hack/e2e-internal/e2e-down.sh"
  )
}

function calculate-node-labels {
  echo "cloud.google.com/metadata-proxy-ready=true"
}
