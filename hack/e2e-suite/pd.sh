#!/bin/bash

# Copyright 2014 Google Inc. All rights reserved.
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

# Launches a container and verifies it can be reached. Assumes that
# we're being called by hack/e2e-test.sh (we use some env vars it sets up).

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
source "${KUBE_ROOT}/cluster/kube-env.sh"
source "${KUBE_ROOT}/cluster/$KUBERNETES_PROVIDER/util.sh"

if [[ "$KUBERNETES_PROVIDER" != "gce" ]] && [[ "$KUBERNETES_PROVIDER" != "gke" ]]; then
  echo "WARNING: Skipping pd.sh for cloud provider: ${KUBERNETES_PROVIDER}."
  exit 0
fi

disk_name="e2e-$(date +%s)"
config="/tmp/${disk_name}.yaml"

function delete_pd_pod() {
  # Delete the pod this should unmount the PD
  ${KUBECFG} delete pods/testpd
  for i in $(seq 1 30); do
    echo "Waiting for pod to be deleted."
    sleep 5
    all_running=0
    for id in $pod_id_list; do
      current_status=$($KUBECFG -template '{{.currentState.status}}' get pods/$id) || true
      if [[ "$current_status" == "Running" ]]; then
        all_running=1
        break
      fi
    done
    if [[ "${all_running}" == 0 ]]; then
      break
    fi
  done
  if [[ "${all_running}" == 1 ]]; then
    echo "Pods did not delete in time"
    exit 1
  fi
}

function teardown() {
  echo "Cleaning up test artifacts"
  delete_pd_pod
  rm -rf ${config}

  # This should really work immediately after the pod is killed, but
  # it doesn't (yet). So let's be resilient to that.
  #
  # TODO: After
  # https://github.com/GoogleCloudPlatform/kubernetes/issues/3437 is
  # fixed, this should be stricter.
  echo "Trying to delete detached pd."
  if ! gcloud compute disks delete --quiet --zone="${ZONE}" "${disk_name}"; then
      echo
      echo "FAILED TO DELETE PD. AGGRESSIVELY DETACHING ${disk_name}."
      echo
      for minion in "${MINION_NAMES[@]}"; do
	  "${GCLOUD}" compute instances detach-disk --quiet --zone="${ZONE}" --disk="${disk_name}" "${minion}" || true
      done
      # This is lame. GCE internals may not finish the actual detach for a little while.
      deleted="false"
      for i in $(seq 1 12); do
	  sleep 5;
	  if gcloud compute disks delete --quiet --zone="${ZONE}" "${disk_name}"; then
	      deleted="true"
	      break
	  fi
      done
      if [[ ${deleted} != "true" ]]; then
	  # At the end of the day, just give up and leak this thing.
	  echo "REALLY FAILED TO DELETE PD. LEAKING ${disk_name}."
      fi
  fi
}

trap "teardown" EXIT

perl -p -e "s/%.*%/${disk_name}/g" ${KUBE_ROOT}/examples/gce-pd/testpd.yaml > ${config}

# Create and format the disk.
"${GCLOUD}" compute disks create --zone="${ZONE}" --size=10GB "${disk_name}"
"${GCLOUD}" compute instances attach-disk --zone="${ZONE}" --disk="${disk_name}" \
  --device-name temp-data "${MASTER_NAME}"
"${GCLOUD}" compute ssh --zone="${ZONE}" "${MASTER_NAME}" --command "sudo rm -rf /mnt/tmp"
"${GCLOUD}" compute ssh --zone="${ZONE}" "${MASTER_NAME}" --command "sudo mkdir -p /mnt/tmp"
"${GCLOUD}" compute ssh --zone="${ZONE}" "${MASTER_NAME}" --command "sudo /usr/share/google/safe_format_and_mount /dev/disk/by-id/google-temp-data /mnt/tmp"
"${GCLOUD}" compute ssh --zone="${ZONE}" "${MASTER_NAME}" --command "sudo umount /mnt/tmp"
"${GCLOUD}" compute instances detach-disk --zone="${ZONE}" --disk "${disk_name}" "${MASTER_NAME}"

# Create a pod that uses the PD
${KUBECFG} -c ${config} create pods

pod_id_list=$($KUBECFG '-template={{range.items}}{{.id}} {{end}}' -l test=testpd list pods)
# Pod turn up on a clean cluster can take a while for the docker image
# pull, and even longer if the PD mount takes a bit.
all_running=0
for i in $(seq 1 30); do
  echo "Waiting for pod to come up."
  sleep 5
  all_running=1
  for id in $pod_id_list; do
    current_status=$($KUBECFG -template '{{.currentState.status}}' get pods/$id) || true
    if [[ "$current_status" != "Running" ]]; then
      all_running=0
      break
    fi
  done
  if [[ "${all_running}" == 1 ]]; then
    break
  fi
done
if [[ "${all_running}" == 0 ]]; then
  echo "Pods did not come up in time"
  exit 1
fi

delete_pd_pod

# This is really horrible, but the PD detach doesn't always complete in time, and
# checking if it is detached via the GCE API is fairly unfriendly.  So we sleep *sigh*
sleep 20


# Recreate the pod, this should re-mount the PD
${KUBECFG} -c ${config} create pods

pod_id_list=$($KUBECFG '-template={{range.items}}{{.id}} {{end}}' -l test=testpd list pods)
# Pod turn up on a clean cluster can take a while for the docker image pull.
all_running=0
for i in $(seq 1 30); do
  echo "Waiting for pod to come up."
  sleep 5
  all_running=1
  for id in $pod_id_list; do
    current_status=$($KUBECFG -template '{{.currentState.status}}' get pods/$id) || true
    if [[ "$current_status" != "Running" ]]; then
      all_running=0
      break
    fi
  done
  if [[ "${all_running}" == 1 ]]; then
    break
  fi
done
if [[ "${all_running}" == 0 ]]; then
  echo "Pods did not come up in time"
  exit 1
fi
