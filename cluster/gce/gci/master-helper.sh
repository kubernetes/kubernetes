#!/usr/bin/env bash

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
  local internal_address=""
  [[ -n ${2:-} ]] && internal_address="${2}"

  write-master-env
  ensure-gci-metadata-files
  # shellcheck disable=SC2153 # 'MASTER_NAME' is assigned by upstream
  create-master-instance-internal "${MASTER_NAME}" "${address}" "${internal_address}"
}

function replicate-master-instance() {
  local existing_master_zone="${1}"
  local existing_master_name="${2}"
  local existing_master_replicas="${3}"

  local kube_env
  kube_env="$(get-metadata "${existing_master_zone}" "${existing_master_name}" kube-env)"
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

  local master_certs
  master_certs="$(get-metadata "${existing_master_zone}" "${existing_master_name}" kube-master-certs)"

  ETCD_APISERVER_CA_KEY="$(echo "${master_certs}" | grep "ETCD_APISERVER_CA_KEY" |  sed "s/^.*: '//" | sed "s/'$//")"
  ETCD_APISERVER_CA_CERT="$(echo "${master_certs}" | grep "ETCD_APISERVER_CA_CERT" |  sed "s/^.*: '//" | sed "s/'$//")"
  create-etcd-apiserver-certs "etcd-${REPLICA_NAME}" "${REPLICA_NAME}" "${ETCD_APISERVER_CA_CERT}" "${ETCD_APISERVER_CA_KEY}"

  master_certs="$(echo "${master_certs}" | grep -v "ETCD_APISERVER_SERVER_KEY")"
  master_certs="$(echo -e "${master_certs}\nETCD_APISERVER_SERVER_KEY: '${ETCD_APISERVER_SERVER_KEY_BASE64}'")"
  master_certs="$(echo "${master_certs}" | grep -v "ETCD_APISERVER_SERVER_CERT")"
  master_certs="$(echo -e "${master_certs}\nETCD_APISERVER_SERVER_CERT: '${ETCD_APISERVER_SERVER_CERT_BASE64}'")"
  master_certs="$(echo "${master_certs}" | grep -v "ETCD_APISERVER_CLIENT_KEY")"
  master_certs="$(echo -e "${master_certs}\nETCD_APISERVER_CLIENT_KEY: '${ETCD_APISERVER_CLIENT_KEY_BASE64}'")"
  master_certs="$(echo "${master_certs}" | grep -v "ETCD_APISERVER_CLIENT_CERT")"
  master_certs="$(echo -e "${master_certs}\nETCD_APISERVER_CLIENT_CERT: '${ETCD_APISERVER_CLIENT_CERT_BASE64}'")"

  echo "${kube_env}" > "${KUBE_TEMP}/master-kube-env.yaml"
  echo "${master_certs}" > "${KUBE_TEMP}/kube-master-certs.yaml"
  get-metadata "${existing_master_zone}" "${existing_master_name}" cluster-name > "${KUBE_TEMP}/cluster-name.txt"
  get-metadata "${existing_master_zone}" "${existing_master_name}" gci-update-strategy > "${KUBE_TEMP}/gci-update.txt"
  get-metadata "${existing_master_zone}" "${existing_master_name}" cluster-location > "${KUBE_TEMP}/cluster-location.txt"

  create-master-instance-internal "${REPLICA_NAME}"
}


# run-gcloud-command runs a given command over ssh with retries.
function run-gcloud-command() {
  local master_name="${1}"
  local zone="${2}"
  local command="${3}"

  local retries=5
  local sleep_sec=10

  local result=""

  for ((i=0; i<retries; i++)); do
    if result=$(gcloud compute ssh "${master_name}" --project "${PROJECT}" --zone "${zone}" --command "${command}" -- -oConnectTimeout=60 2>&1); then
      echo "Successfully executed '${command}' on ${master_name}"
      return 0
    fi

    sleep "${sleep_sec}"
  done
  echo "Failed to execute '${command}' on ${master_name} despite ${retries} attempts" >&2
  echo "Last attempt failed with: ${result}" >&2
  return 1
}


function create-master-instance-internal() {
  local gcloud="gcloud"
  local retries=5
  local sleep_sec=10
  if [[ "${MASTER_SIZE##*-}" -ge 64 ]]; then  # remove everything up to last dash (inclusive)
    # Workaround for #55777
    retries=30
    sleep_sec=60
  fi

  local -r master_name="${1}"
  local -r address="${2:-}"
  local -r internal_address="${3:-}"

  local preemptible_master=""
  if [[ "${PREEMPTIBLE_MASTER:-}" == "true" ]]; then
    preemptible_master="--preemptible --maintenance-policy TERMINATE"
  fi

  local enable_ip_aliases
  if [[ "${NODE_IPAM_MODE:-}" == "CloudAllocator" ]]; then
    enable_ip_aliases=true
  else
    enable_ip_aliases=false
  fi

  local network
  # shellcheck disable=SC2153 # 'NETWORK' is assigned by upstream
  network=$(make-gcloud-network-argument \
    "${NETWORK_PROJECT}" "${REGION}" "${NETWORK}" "${SUBNETWORK:-}" \
    "${address:-}" "${enable_ip_aliases:-}" "${IP_ALIAS_SIZE:-}")

  local metadata="kube-env=${KUBE_TEMP}/master-kube-env.yaml"
  metadata="${metadata},kubelet-config=${KUBE_TEMP}/master-kubelet-config.yaml"
  metadata="${metadata},user-data=${KUBE_ROOT}/cluster/gce/gci/master.yaml"
  metadata="${metadata},configure-sh=${KUBE_ROOT}/cluster/gce/gci/configure.sh"
  metadata="${metadata},cluster-location=${KUBE_TEMP}/cluster-location.txt"
  metadata="${metadata},cluster-name=${KUBE_TEMP}/cluster-name.txt"
  metadata="${metadata},gci-update-strategy=${KUBE_TEMP}/gci-update.txt"
  metadata="${metadata},kube-master-certs=${KUBE_TEMP}/kube-master-certs.yaml"
  metadata="${metadata},cluster-location=${KUBE_TEMP}/cluster-location.txt"
  metadata="${metadata},kube-master-internal-route=${KUBE_ROOT}/cluster/gce/gci/kube-master-internal-route.sh"
  metadata="${metadata},${MASTER_EXTRA_METADATA}"

  local disk="name=${master_name}-pd"
  disk="${disk},device-name=master-pd"
  disk="${disk},mode=rw"
  disk="${disk},boot=no"
  disk="${disk},auto-delete=no"

  for ((i=0; i<retries; i++)); do
    # We expect ZONE to be set and deliberately do not quote preemptible_master
    # and network
    # shellcheck disable=SC2153 disable=SC2086
    if result=$(${gcloud} compute instances create "${master_name}" \
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
      ${MASTER_MIN_CPU_ARCHITECTURE:+"--min-cpu-platform=${MASTER_MIN_CPU_ARCHITECTURE}"} \
      ${preemptible_master} \
      ${network} 2>&1); then
      echo "${result}" >&2

      if [[ -n "${internal_address:-}" ]]; then
        attach-internal-master-ip "${master_name}" "${ZONE}" "${internal_address}"
      fi
      return 0
    else
      echo "${result}" >&2
      if [[ ! "${result}" =~ "try again later" ]]; then
        echo "Failed to create master instance due to non-retryable error" >&2
        return 1
      fi
      sleep $sleep_sec
    fi
  done

  echo "Failed to create master instance despite ${retries} attempts" >&2
  return 1
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
