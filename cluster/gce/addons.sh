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

# A library with any addon related operations during kube-up, kube-down etc.

# Deploys addons to the kubernetes clusters.
function addons::deploy() {
  # Report logging choice (if any).
  if [[ "${ENABLE_NODE_LOGGING-}" == "true" ]]; then
    echo "+++ Logging using Fluentd to ${LOGGING_DESTINATION:-unknown}"
  fi

  # Create autoscaler for nodes if requested
  if [[ "${ENABLE_NODE_AUTOSCALER}" == "true" ]]; then
    METRICS=""
    # Current usage
    METRICS+="--custom-metric-utilization metric=custom.cloudmonitoring.googleapis.com/kubernetes.io/cpu/node_utilization,"
    METRICS+="utilization-target=${TARGET_NODE_UTILIZATION},utilization-target-type=GAUGE "
    METRICS+="--custom-metric-utilization metric=custom.cloudmonitoring.googleapis.com/kubernetes.io/memory/node_utilization,"
    METRICS+="utilization-target=${TARGET_NODE_UTILIZATION},utilization-target-type=GAUGE "

    # Reservation
    METRICS+="--custom-metric-utilization metric=custom.cloudmonitoring.googleapis.com/kubernetes.io/cpu/node_reservation,"
    METRICS+="utilization-target=${TARGET_NODE_UTILIZATION},utilization-target-type=GAUGE "
    METRICS+="--custom-metric-utilization metric=custom.cloudmonitoring.googleapis.com/kubernetes.io/memory/node_reservation,"
    METRICS+="utilization-target=${TARGET_NODE_UTILIZATION},utilization-target-type=GAUGE "

    echo "Creating node autoscalers."

    local max_instances_per_mig=$(((${AUTOSCALER_MAX_NODES} + ${num_migs} - 1) / ${num_migs}))
    local last_max_instances=$((${AUTOSCALER_MAX_NODES} - (${num_migs} - 1) * ${max_instances_per_mig}))
    local min_instances_per_mig=$(((${AUTOSCALER_MIN_NODES} + ${num_migs} - 1) / ${num_migs}))
    local last_min_instances=$((${AUTOSCALER_MIN_NODES} - (${num_migs} - 1) * ${min_instances_per_mig}))

    for i in $(seq $((${num_migs} - 1))); do
      gcloud compute instance-groups managed set-autoscaling "${NODE_INSTANCE_PREFIX}-group-$i" --zone "${ZONE}" --project "${PROJECT}" \
        --min-num-replicas "${min_instances_per_mig}" --max-num-replicas "${max_instances_per_mig}" ${METRICS} || true
    done
    gcloud compute instance-groups managed set-autoscaling "${NODE_INSTANCE_PREFIX}-group" --zone "${ZONE}" --project "${PROJECT}" \
    --min-num-replicas "${last_min_instances}" --max-num-replicas "${last_max_instances}" ${METRICS} || true
  fi
}