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

# Validates that the cluster is healthy.
# Error codes are:
# 0 - success
# 1 - fatal (cluster is unlikely to work)
# 2 - non-fatal (encountered some errors, but cluster should be working correctly)

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..

if [ -f "${KUBE_ROOT}/cluster/env.sh" ]; then
  source "${KUBE_ROOT}/cluster/env.sh"
fi

source "${KUBE_ROOT}/cluster/lib/util.sh"
source "${KUBE_ROOT}/cluster/kube-util.sh"

# Run kubectl and retry upon failure.
function kubectl_retry() {
  tries=3
  while ! "${KUBE_ROOT}/cluster/kubectl.sh" "$@"; do
    tries=$((tries-1))
    if [[ ${tries} -le 0 ]]; then
      echo "('kubectl $@' failed, giving up)" >&2
      return 1
    fi
    echo "(kubectl failed, will retry ${tries} times)" >&2
    sleep 1
  done
}

ALLOWED_NOTREADY_NODES="${ALLOWED_NOTREADY_NODES:-0}"
CLUSTER_READY_ADDITIONAL_TIME_SECONDS="${CLUSTER_READY_ADDITIONAL_TIME_SECONDS:-30}"

EXPECTED_NUM_NODES="${NUM_NODES}"
if [[ "${KUBERNETES_PROVIDER:-}" == "gce" ]]; then
  echo "Validating gce cluster, MULTIZONE=${MULTIZONE:-}"
  # In multizone mode we need to add instances for all nodes in the region.
  if [[ "${MULTIZONE:-}" == "true" ]]; then
    EXPECTED_NUM_NODES=$(gcloud -q compute instances list --project="${PROJECT}" --format=[no-heading] --regexp="${NODE_INSTANCE_PREFIX}.*" \
      --zones=$(gcloud -q compute zones list --project="${PROJECT}" --filter=region=${REGION} --format=[no-heading]\(name\) | tr "\n" "," | sed  "s/,$//") | wc -l)
    echo "Computing number of nodes, NODE_INSTANCE_PREFIX=${NODE_INSTANCE_PREFIX}, REGION=${REGION}, EXPECTED_NUM_NODES=${EXPECTED_NUM_NODES}"
  fi
fi

if [[ "${REGISTER_MASTER_KUBELET:-}" == "true" ]]; then
  if [[ "${KUBERNETES_PROVIDER:-}" == "gce" ]]; then
    NUM_MASTERS=$(get-master-replicas-count)
  else
    NUM_MASTERS=1
  fi
  EXPECTED_NUM_NODES=$((EXPECTED_NUM_NODES+NUM_MASTERS))
fi

REQUIRED_NUM_NODES=$((EXPECTED_NUM_NODES - ALLOWED_NOTREADY_NODES))
# Make several attempts to deal with slow cluster birth.
return_value=0
attempt=0
# Set the timeout to ~25minutes (100 x 15 second) to avoid timeouts for 1000-node clusters.
PAUSE_BETWEEN_ITERATIONS_SECONDS=15
MAX_ATTEMPTS=100
ADDITIONAL_ITERATIONS=$(((CLUSTER_READY_ADDITIONAL_TIME_SECONDS + PAUSE_BETWEEN_ITERATIONS_SECONDS - 1)/PAUSE_BETWEEN_ITERATIONS_SECONDS))
while true; do
  # Pause between iterations of this large outer loop.
  if [[ ${attempt} -gt 0 ]]; then
    sleep 15
  fi
  attempt=$((attempt+1))

  # The "kubectl get nodes -o template" exports node information.
  #
  # Echo the output and gather 2 counts:
  #  - Total number of nodes.
  #  - Number of "ready" nodes.
  #
  # Suppress errors from kubectl output because during cluster bootstrapping
  # for clusters where the master node is registered, the apiserver will become
  # available and then get restarted as the kubelet configures the docker bridge.
  #
  # We are assigning the result of kubectl_retry get nodes operation to the res
  # varaible in that way, to prevent stopping the whole script on an error.
  node=$(kubectl_retry get nodes) && res="$?" || res="$?"
  if [ "${res}" -ne "0" ]; then
    if [[ "${attempt}" -gt "${last_run:-$MAX_ATTEMPTS}" ]]; then
      echo -e "${color_red} Failed to get nodes.${color_norm}"
      exit 1
    else
      continue
    fi
  fi
  found=$(($(echo "${node}" | wc -l) - 1))
  ready=$(($(echo "${node}" | grep -v "NotReady" | wc -l ) - 1))

  if (( "${found}" == "${EXPECTED_NUM_NODES}" )) && (( "${ready}" == "${EXPECTED_NUM_NODES}")); then
    break
  elif (( "${found}" > "${EXPECTED_NUM_NODES}" )); then
    if [[ "${KUBE_USE_EXISTING_MASTER:-}" != "true" ]]; then
      echo -e "${color_red}Found ${found} nodes, but expected ${EXPECTED_NUM_NODES}. Your cluster may not behave correctly.${color_norm}"
    fi
    break
  elif (( "${ready}" > "${EXPECTED_NUM_NODES}")); then
    echo -e "${color_red}Found ${ready} ready nodes, but expected ${EXPECTED_NUM_NODES}. Your cluster may not behave correctly.${color_norm}"
    break
  else
    if [[ "${REQUIRED_NUM_NODES}" -le "${ready}" ]]; then
      echo -e "${color_green}Found ${REQUIRED_NUM_NODES} Nodes, allowing additional ${ADDITIONAL_ITERATIONS} iterations for other Nodes to join.${color_norm}"
      last_run="${last_run:-$((attempt + ADDITIONAL_ITERATIONS - 1))}"
    fi
    if [[ "${attempt}" -gt "${last_run:-$MAX_ATTEMPTS}" ]]; then
      echo -e "${color_yellow}Detected ${ready} ready nodes, found ${found} nodes out of expected ${EXPECTED_NUM_NODES}. Your cluster may not be fully functional.${color_norm}"
      kubectl_retry get nodes
      if [[ "${REQUIRED_NUM_NODES}" -gt "${ready}" ]]; then
        exit 1
      else
        return_value=2
        break
      fi
    else
      echo -e "${color_yellow}Waiting for ${EXPECTED_NUM_NODES} ready nodes. ${ready} ready nodes, ${found} registered. Retrying.${color_norm}"
    fi
  fi
done
echo "Found ${found} node(s)."
kubectl_retry get nodes

attempt=0
while true; do
  # The "kubectl componentstatuses -o template" exports components health information.
  #
  # Echo the output and gather 2 counts:
  #  - Total number of componentstatuses.
  #  - Number of "healthy" components.
  cs_status=$(kubectl_retry get componentstatuses -o template --template='{{range .items}}{{with index .conditions 0}}{{.type}}:{{.status}}{{end}}{{"\n"}}{{end}}') || true
  componentstatuses=$(echo "${cs_status}" | grep -c 'Healthy:') || true
  healthy=$(echo "${cs_status}" | grep -c 'Healthy:True') || true

  if ((componentstatuses > healthy)); then
    if ((attempt < 5)); then
      echo -e "${color_yellow}Cluster not working yet.${color_norm}"
      attempt=$((attempt+1))
      sleep 30
    else
      echo -e " ${color_yellow}Validate output:${color_norm}"
      kubectl_retry get cs
      echo -e "${color_red}Validation returned one or more failed components. Cluster is probably broken.${color_norm}"
      exit 1
    fi
  else
    break
  fi
done

echo "Validate output:"
kubectl_retry get cs
if [ "${return_value}" == "0" ]; then
  echo -e "${color_green}Cluster validation succeeded${color_norm}"
else
  echo -e "${color_yellow}Cluster validation encountered some problems, but cluster should be in working order${color_norm}"
fi

exit "${return_value}"
