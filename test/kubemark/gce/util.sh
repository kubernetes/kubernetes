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

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../../..

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

# This function isn't too robust to race, but that should be ok given its one-off usage during setup.
function get-or-create-master-ip {
  MASTER_IP=$(gcloud compute addresses describe "${MASTER_NAME}-ip" \
    --project "${PROJECT}" --region "${REGION}" -q --format='value(address)') 2>/dev/null || true

  if [[ -z "${MASTER_IP:-}" ]]; then
    run-gcloud-compute-with-retries addresses create "${MASTER_NAME}-ip" \
    --project "${PROJECT}" \
    --region "${REGION}" -q

    MASTER_IP=$(gcloud compute addresses describe "${MASTER_NAME}-ip" \
    --project "${PROJECT}" --region "${REGION}" -q --format='value(address)')
  fi
}

function create-master-instance-with-resources {
  GCLOUD_COMMON_ARGS="--project ${PROJECT} --zone ${ZONE}"
  # Override the master image project to cos-cloud for COS images staring with `cos` string prefix.
  DEFAULT_GCI_PROJECT=google-containers
  if [[ "${GCI_VERSION}" == "cos"* ]]; then
      DEFAULT_GCI_PROJECT=cos-cloud
  fi
  MASTER_IMAGE_PROJECT=${KUBE_GCE_MASTER_PROJECT:-${DEFAULT_GCI_PROJECT}}

  run-gcloud-compute-with-retries disks create "${MASTER_NAME}-pd" \
    ${GCLOUD_COMMON_ARGS} \
    --type "${MASTER_DISK_TYPE}" \
    --size "${MASTER_DISK_SIZE}" &
  
  if [ "${EVENT_PD:-}" == "true" ]; then
    run-gcloud-compute-with-retries disks create "${MASTER_NAME}-event-pd" \
      ${GCLOUD_COMMON_ARGS} \
      --type "${MASTER_DISK_TYPE}" \
      --size "${MASTER_DISK_SIZE}" &
  fi

  get-or-create-master-ip &

  wait

  run-gcloud-compute-with-retries instances create "${MASTER_NAME}" \
    ${GCLOUD_COMMON_ARGS} \
    --address "${MASTER_IP}" \
    --machine-type "${MASTER_SIZE}" \
    --image-project="${MASTER_IMAGE_PROJECT}" \
    --image "${MASTER_IMAGE}" \
    --tags "${MASTER_TAG}" \
    --subnet "${SUBNETWORK:-${NETWORK}}" \
    --scopes "storage-ro,logging-write" \
    --boot-disk-size "${MASTER_ROOT_DISK_SIZE}" \
    --disk "name=${MASTER_NAME}-pd,device-name=master-pd,mode=rw,boot=no,auto-delete=no"

  run-gcloud-compute-with-retries instances add-metadata "${MASTER_NAME}" \
    ${GCLOUD_COMMON_ARGS} \
    --metadata-from-file startup-script="${KUBE_ROOT}/test/kubemark/resources/start-kubemark-master.sh" &
  
  if [ "${EVENT_PD:-}" == "true" ]; then
    echo "Attaching ${MASTER_NAME}-event-pd to ${MASTER_NAME}"
    run-gcloud-compute-with-retries instances attach-disk "${MASTER_NAME}" \
    ${GCLOUD_COMMON_ARGS} \
    --disk "${MASTER_NAME}-event-pd" \
    --device-name="master-event-pd" &
  fi
  
  run-gcloud-compute-with-retries firewall-rules create "${MASTER_NAME}-https" \
    --project "${PROJECT}" \
    --network "${NETWORK}" \
    --source-ranges "0.0.0.0/0" \
    --target-tags "${MASTER_TAG}" \
    --allow "tcp:443" &

  wait
}

# Command to be executed is '$1'.
# No. of retries is '$2' (if provided) or 1 (default).
function execute-cmd-on-master-with-retries() {
  RETRIES="${2:-1}" run-gcloud-compute-with-retries ssh "${MASTER_NAME}" --zone="${ZONE}" --project="${PROJECT}" --command="$1"
}

function copy-files() {
	run-gcloud-compute-with-retries scp --recurse --zone="${ZONE}" --project="${PROJECT}" $@
}

function delete-master-instance-and-resources {
  GCLOUD_COMMON_ARGS="--project ${PROJECT} --zone ${ZONE} --quiet"

  gcloud compute instances delete "${MASTER_NAME}" \
      ${GCLOUD_COMMON_ARGS} || true
  
  gcloud compute disks delete "${MASTER_NAME}-pd" \
      ${GCLOUD_COMMON_ARGS} || true
  
  gcloud compute disks delete "${MASTER_NAME}-event-pd" \
      ${GCLOUD_COMMON_ARGS} &> /dev/null || true
  
  gcloud compute addresses delete "${MASTER_NAME}-ip" \
      --project "${PROJECT}" \
      --region "${REGION}" \
      --quiet || true
  
  gcloud compute firewall-rules delete "${MASTER_NAME}-https" \
	  --project "${PROJECT}" \
	  --quiet || true
  
  if [ "${SEPARATE_EVENT_MACHINE:-false}" == "true" ]; then
	  gcloud compute instances delete "${EVENT_STORE_NAME}" \
    	  ${GCLOUD_COMMON_ARGS} || true
  
	  gcloud compute disks delete "${EVENT_STORE_NAME}-pd" \
    	  ${GCLOUD_COMMON_ARGS} || true
  fi
}
