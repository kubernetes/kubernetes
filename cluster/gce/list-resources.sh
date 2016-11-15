#!/bin/bash

# Copyright 2015 The Kubernetes Authors.
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

# Calls gcloud to print out a variety of Google Cloud Platform resources used by
# Kubernetes. Can be run before/after test runs and compared to track leaking
# resources.

# PROJECT must be set in the environment.
# If ZONE, KUBE_GCE_INSTANCE_PREFIX, CLUSTER_NAME, KUBE_GCE_NETWORK, or
# KUBE_GKE_NETWORK is set, they will be used to filter the results.

set -o errexit
set -o nounset
set -o pipefail

ZONE=${ZONE:-}
REGION=${ZONE%-*}
INSTANCE_PREFIX=${KUBE_GCE_INSTANCE_PREFIX:-${CLUSTER_NAME:-}}
NETWORK=${KUBE_GCE_NETWORK:-${KUBE_GKE_NETWORK:-}}

# In GKE the instance prefix starts with "gke-".
if [[ "${KUBERNETES_PROVIDER:-}" == "gke" ]]; then
  INSTANCE_PREFIX="gke-${CLUSTER_NAME}"
  # Truncate to 26 characters for route prefix matching.
  INSTANCE_PREFIX="${INSTANCE_PREFIX:0:26}"
fi

# Usage: gcloud-compute-list <resource> <additional parameters to gcloud...>
# GREP_REGEX is applied to the output of gcloud if set
GREP_REGEX=""
function gcloud-compute-list() {
  local -r resource=$1
  echo -e "\n\n[ ${resource} ]"
  local attempt=1
  local result=""
  while true; do
    if result=$(gcloud compute ${resource} list --project=${PROJECT} ${@:2}); then
      if [[ ! -z "${GREP_REGEX}" ]]; then
        result=$(echo "${result}" | grep "${GREP_REGEX}")
      fi
      echo "${result}"
      return
    fi
    echo -e "Attempt ${attempt} failed to list ${resource}. Retrying." >&2
    attempt=$(($attempt+1))
    if [[ ${attempt} > 5 ]]; then
      echo -e "List ${resource} failed!" >&2
      exit 2
    fi
    sleep $((5*${attempt}))
  done
}

echo "Project: ${PROJECT}"
echo "Region: ${REGION}"
echo "Zone: ${ZONE}"
echo "Instance prefix: ${INSTANCE_PREFIX:-}"
echo "Network: ${NETWORK}"
echo "Provider: ${KUBERNETES_PROVIDER:-}"

# List resources related to instances, filtering by the instance prefix if
# provided.
gcloud-compute-list instance-templates --regexp="${INSTANCE_PREFIX}.*"
gcloud-compute-list instance-groups ${ZONE:+"--zones=${ZONE}"} --regexp="${INSTANCE_PREFIX}.*"
gcloud-compute-list instances ${ZONE:+"--zones=${ZONE}"} --regexp="${INSTANCE_PREFIX}.*"

# List disk resources, filterying by instance prefix if provided.
gcloud-compute-list disks ${ZONE:+"--zones=${ZONE}"} --regexp="${INSTANCE_PREFIX}.*"

# List network resources. We include names starting with "a", corresponding to
# those that Kubernetes creates.
gcloud-compute-list addresses ${REGION:+"--regions=${REGION}"} --regexp="a.*|${INSTANCE_PREFIX}.*"
# Match either the header or a line with the specified e2e network.
# This assumes that the network name is the second field in the output.
GREP_REGEX="^NAME\|^[^ ]\+[ ]\+\(default\|${NETWORK}\) "
gcloud-compute-list routes --regexp="default.*|${INSTANCE_PREFIX}.*"
gcloud-compute-list firewall-rules --regexp="default.*|k8s-fw.*|${INSTANCE_PREFIX}.*"
GREP_REGEX=""
gcloud-compute-list forwarding-rules ${REGION:+"--regions=${REGION}"}
gcloud-compute-list target-pools ${REGION:+"--regions=${REGION}"}
