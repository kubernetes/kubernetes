#!/bin/bash

# Copyright 2015 The Kubernetes Authors All rights reserved.
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

# A library with any node related operations during kube-up, kube-down etc.

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
source "${KUBE_ROOT}/cluster/gce/common.sh"

if [[ "${OS_DISTRIBUTION}" == "debian" || "${OS_DISTRIBUTION}" == "coreos" || "${OS_DISTRIBUTION}" == "trusty" ]]; then
  source "${KUBE_ROOT}/cluster/gce/${OS_DISTRIBUTION}/helper.sh"
else
  echo "Cannot operate on cluster using os distro: ${OS_DISTRIBUTION}" >&2
  exit 1
fi

# Provisions nodes together with appropriate network configuration. This will also deploy kubernetes
# binaries and start them.
# TODO: We should refactor this to keep provisioning and deploying separate.
function nodes::provision {
  nodes::configure-network

  echo "Creating nodes."

  # TODO(zmerlynn): Refactor setting scope flags.
  local scope_flags=
  if [ -n "${NODE_SCOPES}" ]; then
    scope_flags="--scopes ${NODE_SCOPES}"
  else
    scope_flags="--no-scopes"
  fi

  write-node-env

  local template_name="${NODE_INSTANCE_PREFIX}-template"

  nodes::create-instance-template $template_name

  local defaulted_max_instances_per_mig=${MAX_INSTANCES_PER_MIG:-500}

  if [[ ${defaulted_max_instances_per_mig} -le "0" ]]; then
    echo "MAX_INSTANCES_PER_MIG cannot be negative. Assuming default 500"
    defaulted_max_instances_per_mig=500
  fi
  local num_migs=$(((${NUM_NODES} + ${defaulted_max_instances_per_mig} - 1) / ${defaulted_max_instances_per_mig}))
  local instances_per_mig=$(((${NUM_NODES} + ${num_migs} - 1) / ${num_migs}))
  local last_mig_size=$((${NUM_NODES} - (${num_migs} - 1) * ${instances_per_mig}))

  #TODO: parallelize this loop to speed up the process
  for i in $(seq $((${num_migs} - 1))); do
    gcloud compute instance-groups managed \
      create "${NODE_INSTANCE_PREFIX}-group-$i" \
      --project "${PROJECT}" \
      --zone "${ZONE}" \
      --base-instance-name "${NODE_INSTANCE_PREFIX}" \
      --size "${instances_per_mig}" \
      --template "$template_name" || true;
    gcloud compute instance-groups managed wait-until-stable \
      "${NODE_INSTANCE_PREFIX}-group-$i" \
      --zone "${ZONE}" \
      --project "${PROJECT}" || true;
  done

  # TODO: We don't add a suffix for the last group to keep backward compatibility when there's only one MIG.
  # We should change it at some point, but note #18545 when changing this.
  gcloud compute instance-groups managed \
    create "${NODE_INSTANCE_PREFIX}-group" \
    --project "${PROJECT}" \
    --zone "${ZONE}" \
    --base-instance-name "${NODE_INSTANCE_PREFIX}" \
    --size "${last_mig_size}" \
    --template "$template_name" || true;
  gcloud compute instance-groups managed wait-until-stable \
    "${NODE_INSTANCE_PREFIX}-group" \
    --zone "${ZONE}" \
    --project "${PROJECT}" || true;
}


function nodes::configure-network {
  detect-project

  echo "Creating firewall rule for inter-cluster communication between all nodes"
  create-firewall-rule "${NODE_TAG}-all" "${CLUSTER_IP_RANGE}" "${NODE_TAG}" &
}
