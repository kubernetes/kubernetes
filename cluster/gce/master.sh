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

# A library with any master related operations during kube-up, kube-down etc.

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
source "${KUBE_ROOT}/cluster/gce/common.sh"

# Provision master machine and configures network to give access to it over HTTPS.
function master::provision {
  master::configure-network
  master::create

  # Wait for last batch of jobs
  wait-for-jobs
}

# Configures network to give access to master machine over HTTPS.
function master::configure-network {
  detect-project

  echo "Creating firewall rule for HTTPS traffic to ${MASTER_NAME}"
  gcloud compute firewall-rules create "${MASTER_NAME}-https" \
    --project "${PROJECT}" \
    --network "${NETWORK}" \
    --target-tags "${MASTER_TAG}" \
    --allow tcp:443 &
}

# Creates master machine with a static IP and persistent disk.
function master::create {
  KUBE_MASTER=${MASTER_NAME}
  master::create-disk
  master::create-ip
  master::create-instance "${KUBE_MASTER_IP}" &
}

# Creates persistent disk for master machine.
function master::create-disk {
  detect-project

  # We have to make sure the disk is created before creating the master VM, so
  # run this in the foreground.
  gcloud compute disks create "${MASTER_NAME}-pd" \
    --project "${PROJECT}" \
    --zone "${ZONE}" \
    --type "${MASTER_DISK_TYPE}" \
    --size "${MASTER_DISK_SIZE}"  
}

# Creates static IP address for master machine.
function master::create-ip {
  detect-project

  # Reserve the master's IP so that it can later be transferred to another VM
  # without disrupting the kubelets. IPs are associated with regions, not zones,
  # so extract the region name, which is the same as the zone but with the final
  # dash and characters trailing the dash removed.
  local REGION=${ZONE%-*}

  local attempt=0
  while true; do
    if ! gcloud compute addresses create "${MASTER_NAME}-ip" \
      --project "${PROJECT}" \
      --region "${REGION}" -q > /dev/null; then
      if (( attempt > 4 )); then
        echo -e "${color_red}Failed to create static ip $1 ${color_norm}" >&2
        exit 2
      fi
      attempt=$(($attempt+1)) 
      echo -e "${color_yellow}Attempt $attempt failed to create static ip $1. Retrying.${color_norm}" >&2
      sleep $(($attempt * 5))
    else
      break
    fi
  done

  KUBE_MASTER_IP=$(gcloud compute addresses describe "${MASTER_NAME}-ip" \
    --project "${PROJECT}" \
    --region "${REGION}" -q --format yaml | awk '/^address:/ { print $2 }')
}

# Creates the master instance. If called with an argument, the argument is
# used as the name to a reserved IP address for the master. (In the case of
# upgrade/repair, we re-use the same IP.)
#
# It requires a whole slew of assumed variables, partially due to to
# the call to write-master-env. Listing them would be rather
# futile. Instead, we list the required calls to ensure any additional
# variables are set:
#   ensure-temp-dir
#   detect-project
#   get-bearer-token
#
function master::create-instance {
  detect-project
  local address_opt=""
  [[ -n ${1:-} ]] && address_opt="--address ${1}"

  gcloud compute instances create "${MASTER_NAME}" \
    ${address_opt} \
    --project "${PROJECT}" \
    --zone "${ZONE}" \
    --machine-type "${MASTER_SIZE}" \
    --image-project="${MASTER_IMAGE_PROJECT}" \
    --image "${MASTER_IMAGE}" \
    --tags "${MASTER_TAG}" \
    --network "${NETWORK}" \
    --scopes "storage-ro,compute-rw,monitoring,logging-write" \
    --can-ip-forward \
    --disk "name=${MASTER_NAME}-pd,device-name=master-pd,mode=rw,boot=no,auto-delete=no"
}

# Deploys kubernetes on master machine by setting startup script and restarting the machine.
function master::deploy {
  write-master-env
  gcloud compute instances add-metadata "${MASTER_NAME}" \
    --project "${PROJECT}" \
    --zone "${ZONE}" \
    --metadata-from-file \
      "startup-script=${KUBE_ROOT}/cluster/gce/configure-vm.sh,kube-env=${KUBE_TEMP}/master-kube-env.yaml"
  gcloud compute instances reset "${MASTER_NAME}" \
    --project "${PROJECT}" \
    --zone "${ZONE}"
}
