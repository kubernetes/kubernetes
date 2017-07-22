#!/bin/bash

# Copyright 2016 The Kubernetes Authors.
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

# A library of helper functions and constant for GCI distro
source "${KUBE_ROOT}/cluster/gce/gci/helper.sh"

# create-master-instance creates the master instance. If called with
# an argument, the argument is used as the name to a reserved IP
# address for the master. (In the case of upgrade/repair, we re-use
# the same IP.)
#
# It requires a whole slew of assumed variables, partially due to to
# the call to write-master-env. Listing them would be rather
# futile. Instead, we list the required calls to ensure any additional
#
# variables are set:
#   ensure-temp-dir
#   detect-project
#   get-bearer-token
function create-master-instance {
  local address=""
  [[ -n ${1:-} ]] && address="${1}"

  write-master-env
  ensure-gci-metadata-files
  create-master-instance-internal "${MASTER_NAME}" "${address}"
}

function replicate-master-instance() {
  local existing_master_zone="${1}"
  local existing_master_name="${2}"
  local existing_master_replicas="${3}"

  local kube_env="$(get-metadata "${existing_master_zone}" "${existing_master_name}" kube-env)"
  # Substitute INITIAL_ETCD_CLUSTER to enable etcd clustering.
  kube_env="$(echo "${kube_env}" | grep -v "INITIAL_ETCD_CLUSTER")"
  kube_env="$(echo -e "${kube_env}\nINITIAL_ETCD_CLUSTER: '${existing_master_replicas},${REPLICA_NAME}'")"

  # Substitute INITIAL_ETCD_CLUSTER_STATE
  kube_env="$(echo "${kube_env}" | grep -v "INITIAL_ETCD_CLUSTER_STATE")"
  kube_env="$(echo -e "${kube_env}\nINITIAL_ETCD_CLUSTER_STATE: 'existing'")"

  ETCD_CA_KEY="$(echo "${kube_env}" | grep "ETCD_CA_KEY" |  sed "s/^.*: '//" | sed "s/'$//")"
  ETCD_CA_CERT="$(echo "${kube_env}" | grep "ETCD_CA_CERT" |  sed "s/^.*: '//" | sed "s/'$//")"
  create-etcd-certs "${REPLICA_NAME}" "${ETCD_CA_CERT}" "${ETCD_CA_KEY}"

  kube_env="$(echo "${kube_env}" | grep -v "ETCD_PEER_KEY")"
  kube_env="$(echo -e "${kube_env}\nETCD_PEER_KEY: '${ETCD_PEER_KEY_BASE64}'")"
  kube_env="$(echo "${kube_env}" | grep -v "ETCD_PEER_CERT")"
  kube_env="$(echo -e "${kube_env}\nETCD_PEER_CERT: '${ETCD_PEER_CERT_BASE64}'")"

  echo "${kube_env}" > ${KUBE_TEMP}/master-kube-env.yaml
  get-metadata "${existing_master_zone}" "${existing_master_name}" cluster-name > "${KUBE_TEMP}/cluster-name.txt"
  get-metadata "${existing_master_zone}" "${existing_master_name}" gci-update-strategy > "${KUBE_TEMP}/gci-update.txt"
  get-metadata "${existing_master_zone}" "${existing_master_name}" gci-ensure-gke-docker > "${KUBE_TEMP}/gci-ensure-gke-docker.txt"
  get-metadata "${existing_master_zone}" "${existing_master_name}" gci-docker-version > "${KUBE_TEMP}/gci-docker-version.txt"
  get-metadata "${existing_master_zone}" "${existing_master_name}" kube-master-certs > "${KUBE_TEMP}/kube-master-certs.yaml"

  create-master-instance-internal "${REPLICA_NAME}"
}


function create-master-instance-internal() {
  local gcloud="gcloud"
  if [[ "${ENABLE_IP_ALIASES:-}" == 'true' ]]; then
    gcloud="gcloud beta"
  fi

  local -r master_name="${1}"
  local -r address="${2:-}"

  local preemptible_master=""
  if [[ "${PREEMPTIBLE_MASTER:-}" == "true" ]]; then
    preemptible_master="--preemptible --maintenance-policy TERMINATE"
  fi

  local network=$(make-gcloud-network-argument \
    "${NETWORK}" "${address:-}" \
    "${ENABLE_IP_ALIASES:-}" "${IP_ALIAS_SUBNETWORK:-}" "${IP_ALIAS_SIZE:-}")

  local metadata="kube-env=${KUBE_TEMP}/master-kube-env.yaml"
  metadata="${metadata},user-data=${KUBE_ROOT}/cluster/gce/gci/master.yaml"
  metadata="${metadata},configure-sh=${KUBE_ROOT}/cluster/gce/gci/configure.sh"
  metadata="${metadata},cluster-name=${KUBE_TEMP}/cluster-name.txt"
  metadata="${metadata},gci-update-strategy=${KUBE_TEMP}/gci-update.txt"
  metadata="${metadata},gci-ensure-gke-docker=${KUBE_TEMP}/gci-ensure-gke-docker.txt"
  metadata="${metadata},gci-docker-version=${KUBE_TEMP}/gci-docker-version.txt"
  metadata="${metadata},kube-master-certs=${KUBE_TEMP}/kube-master-certs.yaml"

  local disk="name=${master_name}-pd"
  disk="${disk},device-name=master-pd"
  disk="${disk},mode=rw"
  disk="${disk},boot=no"
  disk="${disk},auto-delete=no"

  ${gcloud} compute instances create "${master_name}" \
    --project "${PROJECT}" \
    --zone "${ZONE}" \
    --machine-type "${MASTER_SIZE}" \
    --image-project="${MASTER_IMAGE_PROJECT}" \
    --image "${MASTER_IMAGE}" \
    --tags "${MASTER_TAG}" \
    --scopes "storage-ro,compute-rw,monitoring,logging-write" \
    --metadata-from-file "${metadata}" \
    --disk "${disk}" \
    --boot-disk-size "${MASTER_ROOT_DISK_SIZE}" \
    ${preemptible_master} \
    ${network}
}

function get-metadata() {
  local zone="${1}"
  local name="${2}"
  local key="${3}"
  gcloud compute ssh "${name}" \
    --project "${PROJECT}" \
    --zone "${zone}" \
    --command "curl \"http://metadata.google.internal/computeMetadata/v1/instance/attributes/${key}\" -H \"Metadata-Flavor: Google\"" 2>/dev/null
}
