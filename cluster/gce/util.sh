#!/bin/bash

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

# A library of helper functions and constant for the local config.

# Use the config file specified in $KUBE_CONFIG_FILE, or default to
# config-default.sh.
KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
source "${KUBE_ROOT}/cluster/gce/${KUBE_CONFIG_FILE-"config-default.sh"}"
source "${KUBE_ROOT}/cluster/common.sh"
source "${KUBE_ROOT}/hack/lib/util.sh"

if [[ "${NODE_OS_DISTRIBUTION}" == "debian" || "${NODE_OS_DISTRIBUTION}" == "container-linux" || "${NODE_OS_DISTRIBUTION}" == "trusty" || "${NODE_OS_DISTRIBUTION}" == "gci" || "${NODE_OS_DISTRIBUTION}" == "ubuntu" ]]; then
  source "${KUBE_ROOT}/cluster/gce/${NODE_OS_DISTRIBUTION}/node-helper.sh"
else
  echo "Cannot operate on cluster using node os distro: ${NODE_OS_DISTRIBUTION}" >&2
  exit 1
fi

if [[ "${MASTER_OS_DISTRIBUTION}" == "container-linux" || "${MASTER_OS_DISTRIBUTION}" == "trusty" || "${MASTER_OS_DISTRIBUTION}" == "gci" || "${MASTER_OS_DISTRIBUTION}" == "ubuntu" ]]; then
  source "${KUBE_ROOT}/cluster/gce/${MASTER_OS_DISTRIBUTION}/master-helper.sh"
else
  echo "Cannot operate on cluster using master os distro: ${MASTER_OS_DISTRIBUTION}" >&2
  exit 1
fi

if [[ "${MASTER_OS_DISTRIBUTION}" == "gci" ]]; then
    DEFAULT_GCI_PROJECT=google-containers
    if [[ "${GCI_VERSION}" == "cos"* ]]; then
        DEFAULT_GCI_PROJECT=cos-cloud
    fi
    MASTER_IMAGE_PROJECT=${KUBE_GCE_MASTER_PROJECT:-${DEFAULT_GCI_PROJECT}}
    # If the master image is not set, we use the latest GCI image.
    # Otherwise, we respect whatever is set by the user.
    MASTER_IMAGE=${KUBE_GCE_MASTER_IMAGE:-${GCI_VERSION}}
elif [[ "${MASTER_OS_DISTRIBUTION}" == "debian" ]]; then
    MASTER_IMAGE=${KUBE_GCE_MASTER_IMAGE:-${CVM_VERSION}}
    MASTER_IMAGE_PROJECT=${KUBE_GCE_MASTER_PROJECT:-google-containers}
fi

# Sets node image based on the specified os distro. Currently this function only
# supports gci and debian.
function set-node-image() {
    if [[ "${NODE_OS_DISTRIBUTION}" == "gci" ]]; then
        DEFAULT_GCI_PROJECT=google-containers
        if [[ "${GCI_VERSION}" == "cos"* ]]; then
            DEFAULT_GCI_PROJECT=cos-cloud
        fi

        # If the node image is not set, we use the latest GCI image.
        # Otherwise, we respect whatever is set by the user.
        NODE_IMAGE=${KUBE_GCE_NODE_IMAGE:-${GCI_VERSION}}
        NODE_IMAGE_PROJECT=${KUBE_GCE_NODE_PROJECT:-${DEFAULT_GCI_PROJECT}}
    elif [[ "${NODE_OS_DISTRIBUTION}" == "debian" ]]; then
        NODE_IMAGE=${KUBE_GCE_NODE_IMAGE:-${CVM_VERSION}}
        NODE_IMAGE_PROJECT=${KUBE_GCE_NODE_PROJECT:-google-containers}
    fi
}

set-node-image

# Verfiy cluster autoscaler configuration.
if [[ "${ENABLE_CLUSTER_AUTOSCALER}" == "true" ]]; then
  if [[ -z $AUTOSCALER_MIN_NODES ]]; then
    echo "AUTOSCALER_MIN_NODES not set."
    exit 1
  fi
  if [[ -z $AUTOSCALER_MAX_NODES ]]; then
    echo "AUTOSCALER_MAX_NODES not set."
    exit 1
  fi
fi

NODE_INSTANCE_PREFIX="${INSTANCE_PREFIX}-minion"
NODE_TAGS="${NODE_TAG}"

ALLOCATE_NODE_CIDRS=true
PREEXISTING_NETWORK=false
PREEXISTING_NETWORK_MODE=""

KUBE_PROMPT_FOR_UPDATE=${KUBE_PROMPT_FOR_UPDATE:-"n"}
# How long (in seconds) to wait for cluster initialization.
KUBE_CLUSTER_INITIALIZATION_TIMEOUT=${KUBE_CLUSTER_INITIALIZATION_TIMEOUT:-300}

function join_csv() {
  local IFS=','; echo "$*";
}

# This function returns the first string before the comma
function split_csv() {
  echo "$*" | cut -d',' -f1
}

# Verify prereqs
function verify-prereqs() {
  local cmd
  for cmd in gcloud gsutil; do
    if ! which "${cmd}" >/dev/null; then
      local resp="n"
      if [[ "${KUBE_PROMPT_FOR_UPDATE}" == "y" ]]; then
        echo "Can't find ${cmd} in PATH.  Do you wish to install the Google Cloud SDK? [Y/n]"
        read resp
      fi
      if [[ "${resp}" != "n" && "${resp}" != "N" ]]; then
        curl https://sdk.cloud.google.com | bash
      fi
      if ! which "${cmd}" >/dev/null; then
        echo "Can't find ${cmd} in PATH, please fix and retry. The Google Cloud " >&2
        echo "SDK can be downloaded from https://cloud.google.com/sdk/." >&2
        exit 1
      fi
    fi
  done
  update-or-verify-gcloud
}

# Use the gcloud defaults to find the project.  If it is already set in the
# environment then go with that.
#
# Vars set:
#   PROJECT
#   PROJECT_REPORTED
function detect-project() {
  if [[ -z "${PROJECT-}" ]]; then
    PROJECT=$(gcloud config list project --format 'value(core.project)')
  fi

  if [[ -z "${PROJECT-}" ]]; then
    echo "Could not detect Google Cloud Platform project.  Set the default project using " >&2
    echo "'gcloud config set project <PROJECT>'" >&2
    exit 1
  fi
  if [[ -z "${PROJECT_REPORTED-}" ]]; then
    echo "Project: ${PROJECT}" >&2
    echo "Zone: ${ZONE}" >&2
    PROJECT_REPORTED=true
  fi
}

# Copy a release tar and its accompanying hash.
function copy-to-staging() {
  local -r staging_path=$1
  local -r gs_url=$2
  local -r tar=$3
  local -r hash=$4

  echo "${hash}" > "${tar}.sha1"
  gsutil -m -q -h "Cache-Control:private, max-age=0" cp "${tar}" "${tar}.sha1" "${staging_path}"
  gsutil -m acl ch -g all:R "${gs_url}" "${gs_url}.sha1" >/dev/null 2>&1
  echo "+++ $(basename ${tar}) uploaded (sha1 = ${hash})"
}

# Given the cluster zone, return the list of regional GCS release
# bucket suffixes for the release in preference order. GCS doesn't
# give us an API for this, so we hardcode it.
#
# Assumed vars:
#   RELEASE_REGION_FALLBACK
#   REGIONAL_KUBE_ADDONS
#   ZONE
# Vars set:
#   PREFERRED_REGION
#   KUBE_ADDON_REGISTRY
function set-preferred-region() {
  case ${ZONE} in
    asia-*)
      PREFERRED_REGION=("asia" "us" "eu")
      ;;
    europe-*)
      PREFERRED_REGION=("eu" "us" "asia")
      ;;
    *)
      PREFERRED_REGION=("us" "eu" "asia")
      ;;
  esac
  local -r preferred="${PREFERRED_REGION[0]}"

  if [[ "${RELEASE_REGION_FALLBACK}" != "true" ]]; then
    PREFERRED_REGION=( "${preferred}" )
  fi

  # If we're using regional GCR, and we're outside the US, go to the
  # regional registry. The gcr.io/google_containers registry is
  # appropriate for US (for now).
  if [[ "${REGIONAL_KUBE_ADDONS}" == "true" ]] && [[ "${preferred}" != "us" ]]; then
    KUBE_ADDON_REGISTRY="${preferred}.gcr.io/google_containers"
  else
    KUBE_ADDON_REGISTRY="gcr.io/google_containers"
  fi

  if [[ "${ENABLE_DOCKER_REGISTRY_CACHE:-}" == "true" ]]; then
    DOCKER_REGISTRY_MIRROR_URL="https://${preferred}-mirror.gcr.io"
  fi
}

# Take the local tar files and upload them to Google Storage.  They will then be
# downloaded by the master as part of the start up script for the master.
#
# Assumed vars:
#   PROJECT
#   SERVER_BINARY_TAR
#   SALT_TAR
#   KUBE_MANIFESTS_TAR
#   ZONE
# Vars set:
#   SERVER_BINARY_TAR_URL
#   SERVER_BINARY_TAR_HASH
#   SALT_TAR_URL
#   SALT_TAR_HASH
#   KUBE_MANIFESTS_TAR_URL
#   KUBE_MANIFESTS_TAR_HASH
function upload-server-tars() {
  SERVER_BINARY_TAR_URL=
  SERVER_BINARY_TAR_HASH=
  SALT_TAR_URL=
  SALT_TAR_HASH=
  KUBE_MANIFESTS_TAR_URL=
  KUBE_MANIFESTS_TAR_HASH=

  local project_hash
  if which md5 > /dev/null 2>&1; then
    project_hash=$(md5 -q -s "$PROJECT")
  else
    project_hash=$(echo -n "$PROJECT" | md5sum | awk '{ print $1 }')
  fi

  # This requires 1 million projects before the probability of collision is 50%
  # that's probably good enough for now :P
  project_hash=${project_hash:0:10}

  set-preferred-region

  SERVER_BINARY_TAR_HASH=$(sha1sum-file "${SERVER_BINARY_TAR}")
  SALT_TAR_HASH=$(sha1sum-file "${SALT_TAR}")
  if [[ -n "${KUBE_MANIFESTS_TAR:-}" ]]; then
    KUBE_MANIFESTS_TAR_HASH=$(sha1sum-file "${KUBE_MANIFESTS_TAR}")
  fi

  local server_binary_tar_urls=()
  local salt_tar_urls=()
  local kube_manifest_tar_urls=()

  for region in "${PREFERRED_REGION[@]}"; do
    suffix="-${region}"
    if [[ "${suffix}" == "-us" ]]; then
      suffix=""
    fi
    local staging_bucket="gs://kubernetes-staging-${project_hash}${suffix}"

    # Ensure the buckets are created
    if ! gsutil ls "${staging_bucket}" >/dev/null; then
      echo "Creating ${staging_bucket}"
      gsutil mb -l "${region}" "${staging_bucket}"
    fi

    local staging_path="${staging_bucket}/${INSTANCE_PREFIX}-devel"

    echo "+++ Staging server tars to Google Storage: ${staging_path}"
    local server_binary_gs_url="${staging_path}/${SERVER_BINARY_TAR##*/}"
    local salt_gs_url="${staging_path}/${SALT_TAR##*/}"
    copy-to-staging "${staging_path}" "${server_binary_gs_url}" "${SERVER_BINARY_TAR}" "${SERVER_BINARY_TAR_HASH}"
    copy-to-staging "${staging_path}" "${salt_gs_url}" "${SALT_TAR}" "${SALT_TAR_HASH}"

    # Convert from gs:// URL to an https:// URL
    server_binary_tar_urls+=("${server_binary_gs_url/gs:\/\//https://storage.googleapis.com/}")
    salt_tar_urls+=("${salt_gs_url/gs:\/\//https://storage.googleapis.com/}")
    if [[ -n "${KUBE_MANIFESTS_TAR:-}" ]]; then
      local kube_manifests_gs_url="${staging_path}/${KUBE_MANIFESTS_TAR##*/}"
      copy-to-staging "${staging_path}" "${kube_manifests_gs_url}" "${KUBE_MANIFESTS_TAR}" "${KUBE_MANIFESTS_TAR_HASH}"
      # Convert from gs:// URL to an https:// URL
      kube_manifests_tar_urls+=("${kube_manifests_gs_url/gs:\/\//https://storage.googleapis.com/}")
    fi
  done

  SERVER_BINARY_TAR_URL=$(join_csv "${server_binary_tar_urls[@]}")
  SALT_TAR_URL=$(join_csv "${salt_tar_urls[@]}")
  if [[ -n "${KUBE_MANIFESTS_TAR:-}" ]]; then
    KUBE_MANIFESTS_TAR_URL=$(join_csv "${kube_manifests_tar_urls[@]}")
  fi
}

# Detect minions created in the minion group
#
# Assumed vars:
#   NODE_INSTANCE_PREFIX
# Vars set:
#   NODE_NAMES
#   INSTANCE_GROUPS
function detect-node-names() {
  detect-project
  INSTANCE_GROUPS=()
  INSTANCE_GROUPS+=($(gcloud compute instance-groups managed list \
    --zones "${ZONE}" --project "${PROJECT}" \
    --regexp "${NODE_INSTANCE_PREFIX}-.+" \
    --format='value(instanceGroup)' || true))
  NODE_NAMES=()
  if [[ -n "${INSTANCE_GROUPS[@]:-}" ]]; then
    for group in "${INSTANCE_GROUPS[@]}"; do
      NODE_NAMES+=($(gcloud compute instance-groups managed list-instances \
        "${group}" --zone "${ZONE}" --project "${PROJECT}" \
        --format='value(instance)'))
    done
  fi
  echo "INSTANCE_GROUPS=${INSTANCE_GROUPS[*]:-}" >&2
  echo "NODE_NAMES=${NODE_NAMES[*]:-}" >&2
}

# Detect the information about the minions
#
# Assumed vars:
#   ZONE
# Vars set:
#   NODE_NAMES
#   KUBE_NODE_IP_ADDRESSES (array)
function detect-nodes() {
  detect-project
  detect-node-names
  KUBE_NODE_IP_ADDRESSES=()
  for (( i=0; i<${#NODE_NAMES[@]}; i++)); do
    local node_ip=$(gcloud compute instances describe --project "${PROJECT}" --zone "${ZONE}" \
      "${NODE_NAMES[$i]}" --format='value(networkInterfaces[0].accessConfigs[0].natIP)')
    if [[ -z "${node_ip-}" ]] ; then
      echo "Did not find ${NODE_NAMES[$i]}" >&2
    else
      echo "Found ${NODE_NAMES[$i]} at ${node_ip}"
      KUBE_NODE_IP_ADDRESSES+=("${node_ip}")
    fi
  done
  if [[ -z "${KUBE_NODE_IP_ADDRESSES-}" ]]; then
    echo "Could not detect Kubernetes minion nodes.  Make sure you've launched a cluster with 'kube-up.sh'" >&2
    exit 1
  fi
}

# Detect the IP for the master
#
# Assumed vars:
#   MASTER_NAME
#   ZONE
#   REGION
# Vars set:
#   KUBE_MASTER
#   KUBE_MASTER_IP
function detect-master() {
  detect-project
  KUBE_MASTER=${MASTER_NAME}
  echo "Trying to find master named '${MASTER_NAME}'" >&2
  if [[ -z "${KUBE_MASTER_IP-}" ]]; then
    local master_address_name="${MASTER_NAME}-ip"
    echo "Looking for address '${master_address_name}'" >&2
    KUBE_MASTER_IP=$(gcloud compute addresses describe "${master_address_name}" \
      --project "${PROJECT}" --region "${REGION}" -q --format='value(address)')
  fi
  if [[ -z "${KUBE_MASTER_IP-}" ]]; then
    echo "Could not detect Kubernetes master node.  Make sure you've launched a cluster with 'kube-up.sh'" >&2
    exit 1
  fi
  echo "Using master: $KUBE_MASTER (external IP: $KUBE_MASTER_IP)" >&2
}

# Reads kube-env metadata from master
#
# Assumed vars:
#   KUBE_MASTER
#   PROJECT
#   ZONE
function get-master-env() {
  # TODO(zmerlynn): Make this more reliable with retries.
  gcloud compute --project ${PROJECT} ssh --zone ${ZONE} ${KUBE_MASTER} --command \
    "curl --fail --silent -H 'Metadata-Flavor: Google' \
      'http://metadata/computeMetadata/v1/instance/attributes/kube-env'" 2>/dev/null
  gcloud compute --project ${PROJECT} ssh --zone ${ZONE} ${KUBE_MASTER} --command \
    "curl --fail --silent -H 'Metadata-Flavor: Google' \
      'http://metadata/computeMetadata/v1/instance/attributes/kube-master-certs'" 2>/dev/null
}

# Robustly try to create a static ip.
# $1: The name of the ip to create
# $2: The name of the region to create the ip in.
function create-static-ip() {
  detect-project
  local attempt=0
  local REGION="$2"
  while true; do
    if gcloud compute addresses create "$1" \
      --project "${PROJECT}" \
      --region "${REGION}" -q > /dev/null; then
      # successful operation - wait until it's visible
      start="$(date +%s)"
      while true; do
        now="$(date +%s)"
        # Timeout set to 15 minutes
        if [[ $((now - start)) -gt 900 ]]; then
          echo "Timeout while waiting for master IP visibility"
          exit 2
        fi
        if gcloud compute addresses describe "$1" --project "${PROJECT}" --region "${REGION}" >/dev/null 2>&1; then
          break
        fi
        echo "Master IP not visible yet. Waiting..."
        sleep 5
      done
      break
    fi

    if gcloud compute addresses describe "$1" \
      --project "${PROJECT}" \
      --region "${REGION}" >/dev/null 2>&1; then
      # it exists - postcondition satisfied
      break
    fi

    if (( attempt > 4 )); then
      echo -e "${color_red}Failed to create static ip $1 ${color_norm}" >&2
      exit 2
    fi
    attempt=$(($attempt+1))
    echo -e "${color_yellow}Attempt $attempt failed to create static ip $1. Retrying.${color_norm}" >&2
    sleep $(($attempt * 5))
  done
}

# Robustly try to create a firewall rule.
# $1: The name of firewall rule.
# $2: IP ranges.
# $3: Target tags for this firewall rule.
function create-firewall-rule() {
  detect-project
  local attempt=0
  while true; do
    if ! gcloud compute firewall-rules create "$1" \
      --project "${PROJECT}" \
      --network "${NETWORK}" \
      --source-ranges "$2" \
      --target-tags "$3" \
      --allow tcp,udp,icmp,esp,ah,sctp; then
      if (( attempt > 4 )); then
        echo -e "${color_red}Failed to create firewall rule $1 ${color_norm}" >&2
        exit 2
      fi
      echo -e "${color_yellow}Attempt $(($attempt+1)) failed to create firewall rule $1. Retrying.${color_norm}" >&2
      attempt=$(($attempt+1))
      sleep $(($attempt * 5))
    else
        break
    fi
  done
}

# Format the string argument for gcloud network.
function make-gcloud-network-argument() {
  local network="$1"
  local address="$2"          # optional
  local enable_ip_alias="$3"  # optional
  local alias_subnetwork="$4" # optional
  local alias_size="$5"       # optional

  local ret=""

  if [[ "${enable_ip_alias}" == 'true' ]]; then
    ret="--network-interface"
    ret="${ret} network=${network}"
    # If address is omitted, instance will not receive an external IP.
    ret="${ret},address=${address:-}"
    ret="${ret},subnet=${alias_subnetwork}"
    ret="${ret},aliases=pods-default:${alias_size}"
    ret="${ret} --no-can-ip-forward"
  else
    if [[ ${ENABLE_BIG_CLUSTER_SUBNETS} != "true" || (${PREEXISTING_NETWORK} = "true" && "${PREEXISTING_NETWORK_MODE}" != "custom") ]]; then
      ret="--network ${network}"
    else
      ret="--subnet=${network}"
    fi
    ret="${ret} --can-ip-forward"
    if [[ -n ${address:-} ]]; then
      ret="${ret} --address ${address}"
    fi
  fi

  echo "${ret}"
}

# $1: version (required)
function get-template-name-from-version() {
  # trim template name to pass gce name validation
  echo "${NODE_INSTANCE_PREFIX}-template-${1}" | cut -c 1-63 | sed 's/[\.\+]/-/g;s/-*$//g'
}

# Robustly try to create an instance template.
# $1: The name of the instance template.
# $2: The scopes flag.
# $3 and others: Metadata entries (must all be from a file).
function create-node-template() {
  detect-project
  local template_name="$1"

  # First, ensure the template doesn't exist.
  # TODO(zmerlynn): To make this really robust, we need to parse the output and
  #                 add retries. Just relying on a non-zero exit code doesn't
  #                 distinguish an ephemeral failed call from a "not-exists".
  if gcloud compute instance-templates describe "$template_name" --project "${PROJECT}" &>/dev/null; then
    echo "Instance template ${1} already exists; deleting." >&2
    if ! gcloud compute instance-templates delete "$template_name" --project "${PROJECT}" --quiet &>/dev/null; then
      echo -e "${color_yellow}Failed to delete existing instance template${color_norm}" >&2
      exit 2
    fi
  fi

  local gcloud="gcloud"

  local accelerator_args=""
  # VMs with Accelerators cannot be live migrated.
  # More details here - https://cloud.google.com/compute/docs/gpus/add-gpus#create-new-gpu-instance
  if [[ ! -z "${NODE_ACCELERATORS}" ]]; then
    accelerator_args="--maintenance-policy TERMINATE --restart-on-failure --accelerator ${NODE_ACCELERATORS}"
    gcloud="gcloud beta"
  fi

  if [[ "${ENABLE_IP_ALIASES:-}" == 'true' ]]; then
    gcloud="gcloud beta"
  fi

  local preemptible_minions=""
  if [[ "${PREEMPTIBLE_NODE}" == "true" ]]; then
    preemptible_minions="--preemptible --maintenance-policy TERMINATE"
  fi

  local local_ssds=""
  if [[ ! -z ${NODE_LOCAL_SSDS+x} ]]; then
      for i in $(seq ${NODE_LOCAL_SSDS}); do
          local_ssds="$local_ssds--local-ssd=interface=SCSI "
      done
  fi

  local network=$(make-gcloud-network-argument \
    "${NETWORK}" "" \
    "${ENABLE_IP_ALIASES:-}" \
    "${IP_ALIAS_SUBNETWORK:-}" \
    "${IP_ALIAS_SIZE:-}")

  local attempt=1
  while true; do
    echo "Attempt ${attempt} to create ${1}" >&2
    if ! ${gcloud} compute instance-templates create \
      "$template_name" \
      --project "${PROJECT}" \
      --machine-type "${NODE_SIZE}" \
      --boot-disk-type "${NODE_DISK_TYPE}" \
      --boot-disk-size "${NODE_DISK_SIZE}" \
      --image-project="${NODE_IMAGE_PROJECT}" \
      --image "${NODE_IMAGE}" \
      --tags "${NODE_TAG}" \
      ${accelerator_args} \
      ${local_ssds} \
      --region "${REGION}" \
      ${network} \
      ${preemptible_minions} \
      $2 \
      --metadata-from-file $(echo ${@:3} | tr ' ' ',') >&2; then
        if (( attempt > 5 )); then
          echo -e "${color_red}Failed to create instance template $template_name ${color_norm}" >&2
          exit 2
        fi
        echo -e "${color_yellow}Attempt ${attempt} failed to create instance template $template_name. Retrying.${color_norm}" >&2
        attempt=$(($attempt+1))
        sleep $(($attempt * 5))

        # In case the previous attempt failed with something like a
        # Backend Error and left the entry laying around, delete it
        # before we try again.
        gcloud compute instance-templates delete "$template_name" --project "${PROJECT}" &>/dev/null || true
    else
        break
    fi
  done
}

# Robustly try to add metadata on an instance.
# $1: The name of the instance.
# $2...$n: The metadata key=value pairs to add.
function add-instance-metadata() {
  local -r instance=$1
  shift 1
  local -r kvs=( "$@" )
  detect-project
  local attempt=0
  while true; do
    if ! gcloud compute instances add-metadata "${instance}" \
      --project "${PROJECT}" \
      --zone "${ZONE}" \
      --metadata "${kvs[@]}"; then
        if (( attempt > 5 )); then
          echo -e "${color_red}Failed to add instance metadata in ${instance} ${color_norm}" >&2
          exit 2
        fi
        echo -e "${color_yellow}Attempt $(($attempt+1)) failed to add metadata in ${instance}. Retrying.${color_norm}" >&2
        attempt=$(($attempt+1))
        sleep $((5 * $attempt))
    else
        break
    fi
  done
}

# Robustly try to add metadata on an instance, from a file.
# $1: The name of the instance.
# $2...$n: The metadata key=file pairs to add.
function add-instance-metadata-from-file() {
  local -r instance=$1
  shift 1
  local -r kvs=( "$@" )
  detect-project
  local attempt=0
  while true; do
    echo "${kvs[@]}"
    if ! gcloud compute instances add-metadata "${instance}" \
      --project "${PROJECT}" \
      --zone "${ZONE}" \
      --metadata-from-file "$(join_csv ${kvs[@]})"; then
        if (( attempt > 5 )); then
          echo -e "${color_red}Failed to add instance metadata in ${instance} ${color_norm}" >&2
          exit 2
        fi
        echo -e "${color_yellow}Attempt $(($attempt+1)) failed to add metadata in ${instance}. Retrying.${color_norm}" >&2
        attempt=$(($attempt+1))
        sleep $(($attempt * 5))
    else
        break
    fi
  done
}

# Instantiate a kubernetes cluster
#
# Assumed vars
#   KUBE_ROOT
#   <Various vars set in config file>
function kube-up() {
  kube::util::ensure-temp-dir
  detect-project

  load-or-gen-kube-basicauth
  load-or-gen-kube-bearertoken

  # Make sure we have the tar files staged on Google Storage
  find-release-tars
  upload-server-tars

  # ensure that environmental variables specifying number of migs to create
  set_num_migs

  if [[ ${KUBE_USE_EXISTING_MASTER:-} == "true" ]]; then
    detect-master
    parse-master-env
    create-subnetworks
    create-nodes
  elif [[ ${KUBE_REPLICATE_EXISTING_MASTER:-} == "true" ]]; then
    if  [[ "${MASTER_OS_DISTRIBUTION}" != "gci" && "${MASTER_OS_DISTRIBUTION}" != "debian" && "${MASTER_OS_DISTRIBUTION}" != "ubuntu" ]]; then
      echo "Master replication supported only for gci, debian, and ubuntu"
      return 1
    fi
    create-loadbalancer
    # If replication of master fails, we need to ensure that the replica is removed from etcd clusters.
    if ! replicate-master; then
      remove-replica-from-etcd 2379 || true
      remove-replica-from-etcd 4002 || true
    fi
  else
    check-existing
    create-network
    create-subnetworks
    write-cluster-name
    create-autoscaler-config
    create-master
    create-nodes-firewall
    create-nodes-template
    create-nodes
    check-cluster
  fi
}

function check-existing() {
  local running_in_terminal=false
  # May be false if tty is not allocated (for example with ssh -T).
  if [[ -t 1 ]]; then
    running_in_terminal=true
  fi

  if [[ ${running_in_terminal} == "true" || ${KUBE_UP_AUTOMATIC_CLEANUP} == "true" ]]; then
    if ! check-resources; then
      local run_kube_down="n"
      echo "${KUBE_RESOURCE_FOUND} found." >&2
      # Get user input only if running in terminal.
      if [[ ${running_in_terminal} == "true" && ${KUBE_UP_AUTOMATIC_CLEANUP} == "false" ]]; then
        read -p "Would you like to shut down the old cluster (call kube-down)? [y/N] " run_kube_down
      fi
      if [[ ${run_kube_down} == "y" || ${run_kube_down} == "Y" || ${KUBE_UP_AUTOMATIC_CLEANUP} == "true" ]]; then
        echo "... calling kube-down" >&2
        kube-down
      fi
    fi
  fi
}

function create-network() {
  if ! gcloud compute networks --project "${PROJECT}" describe "${NETWORK}" &>/dev/null; then
    echo "Creating new network: ${NETWORK}"
    # The network needs to be created synchronously or we have a race. The
    # firewalls can be added concurrent with instance creation.
    gcloud compute networks create --project "${PROJECT}" "${NETWORK}" --mode=auto
  else
    PREEXISTING_NETWORK=true
    PREEXISTING_NETWORK_MODE="$(gcloud compute networks list ${NETWORK} --format='value(x_gcloud_mode)' || true)"
    echo "Found existing network ${NETWORK} in ${PREEXISTING_NETWORK_MODE} mode."
  fi

  if ! gcloud compute firewall-rules --project "${PROJECT}" describe "${CLUSTER_NAME}-default-internal-master" &>/dev/null; then
    gcloud compute firewall-rules create "${CLUSTER_NAME}-default-internal-master" \
      --project "${PROJECT}" \
      --network "${NETWORK}" \
      --source-ranges "10.0.0.0/8" \
      --allow "tcp:1-2379,tcp:2382-65535,udp:1-65535,icmp" \
      --target-tags "${MASTER_TAG}"&
  fi

  if ! gcloud compute firewall-rules --project "${PROJECT}" describe "${CLUSTER_NAME}-default-internal-node" &>/dev/null; then
    gcloud compute firewall-rules create "${CLUSTER_NAME}-default-internal-node" \
      --project "${PROJECT}" \
      --network "${NETWORK}" \
      --source-ranges "10.0.0.0/8" \
      --allow "tcp:1-65535,udp:1-65535,icmp" \
      --target-tags "${NODE_TAG}"&
  fi

  if ! gcloud compute firewall-rules describe --project "${PROJECT}" "${NETWORK}-default-ssh" &>/dev/null; then
    gcloud compute firewall-rules create "${NETWORK}-default-ssh" \
      --project "${PROJECT}" \
      --network "${NETWORK}" \
      --source-ranges "0.0.0.0/0" \
      --allow "tcp:22" &
  fi
}

function expand-default-subnetwork() {
  gcloud compute networks switch-mode "${NETWORK}" \
    --mode custom \
    --project "${PROJECT}" \
    --quiet || true
  gcloud compute networks subnets expand-ip-range "${NETWORK}" \
    --region="${REGION}" \
    --project "${PROJECT}" \
    --prefix-length=19 \
    --quiet
}

function create-subnetworks() {
  case ${ENABLE_IP_ALIASES} in
    true) echo "IP aliases are enabled. Creating subnetworks.";;
    false)
      echo "IP aliases are disabled."
      if [[ "${ENABLE_BIG_CLUSTER_SUBNETS}" = "true" ]]; then 
        if [[  "${PREEXISTING_NETWORK}" != "true" ]]; then
          expand-default-subnetwork
        else
          echo "${color_yellow}Using pre-existing network ${NETWORK}, subnets won't be expanded to /19!${color_norm}"
        fi
      fi
      return;;
    *) echo "${color_red}Invalid argument to ENABLE_IP_ALIASES${color_norm}"
       exit 1;;
  esac

  # Look for the alias subnet, it must exist and have a secondary
  # range configured.
  local subnet=$(gcloud beta compute networks subnets describe \
    --project "${PROJECT}" \
    --region ${REGION} \
    ${IP_ALIAS_SUBNETWORK} 2>/dev/null)
  if [[ -z ${subnet} ]]; then
    # Only allow auto-creation for default subnets
    if [[ ${IP_ALIAS_SUBNETWORK} != ${INSTANCE_PREFIX}-subnet-default ]]; then
      echo "${color_red}Subnetwork ${NETWORK}:${IP_ALIAS_SUBNETWORK} does not exist${color_norm}"
      exit 1
    fi

    if [[ -z ${NODE_IP_RANGE:-} ]]; then
      echo "${color_red}NODE_IP_RANGE must be specified{color_norm}"
      exit 1
    fi

    echo "Creating subnet ${NETWORK}:${IP_ALIAS_SUBNETWORK}"
    gcloud beta compute networks subnets create \
      ${IP_ALIAS_SUBNETWORK} \
      --description "Automatically generated subnet for ${INSTANCE_PREFIX} cluster. This will be removed on cluster teardown." \
      --project "${PROJECT}" \
      --network ${NETWORK} \
      --region ${REGION} \
      --range ${NODE_IP_RANGE} \
      --secondary-range "pods-default=${CLUSTER_IP_RANGE}"
    echo "Created subnetwork ${IP_ALIAS_SUBNETWORK}"
  else
    if ! echo ${subnet} | grep --quiet secondaryIpRanges ${subnet}; then
      echo "${color_red}Subnet ${IP_ALIAS_SUBNETWORK} does not have a secondary range${color_norm}"
      exit 1
    fi
  fi

  # Services subnetwork.
  local subnet=$(gcloud beta compute networks subnets describe \
    --project "${PROJECT}" \
    --region ${REGION} \
    ${SERVICE_CLUSTER_IP_SUBNETWORK} 2>/dev/null)

  if [[ -z ${subnet} ]]; then
    if [[ ${SERVICE_CLUSTER_IP_SUBNETWORK} != ${INSTANCE_PREFIX}-subnet-services ]]; then
      echo "${color_red}Subnetwork ${NETWORK}:${SERVICE_CLUSTER_IP_SUBNETWORK} does not exist${color_norm}"
      exit 1
    fi

    echo "Creating subnet for reserving service cluster IPs ${NETWORK}:${SERVICE_CLUSTER_IP_SUBNETWORK}"
    gcloud beta compute networks subnets create \
      ${SERVICE_CLUSTER_IP_SUBNETWORK} \
      --description "Automatically generated subnet for ${INSTANCE_PREFIX} cluster. This will be removed on cluster teardown." \
      --project "${PROJECT}" \
      --network ${NETWORK} \
      --region ${REGION} \
      --range ${SERVICE_CLUSTER_IP_RANGE}
    echo "Created subnetwork ${SERVICE_CLUSTER_IP_SUBNETWORK}"
  else
    echo "Subnet ${SERVICE_CLUSTER_IP_SUBNETWORK} already exists"
  fi
}

function delete-firewall-rules() {
  for fw in $@; do
    if [[ -n $(gcloud compute firewall-rules --project "${PROJECT}" describe "${fw}" --format='value(name)' 2>/dev/null || true) ]]; then
      gcloud compute firewall-rules delete --project "${PROJECT}" --quiet "${fw}" &
    fi
  done
  kube::util::wait-for-jobs || {
    echo -e "${color_red}Failed to delete firewall rules.${color_norm}" >&2
  }
}

function delete-network() {
  if [[ -n $(gcloud compute networks --project "${PROJECT}" describe "${NETWORK}" --format='value(name)' 2>/dev/null || true) ]]; then
    if ! gcloud compute networks delete --project "${PROJECT}" --quiet "${NETWORK}"; then
      echo "Failed to delete network '${NETWORK}'. Listing firewall-rules:"
      gcloud compute firewall-rules --project "${PROJECT}" list --filter="network=${NETWORK}"
      return 1
    fi
  fi
}

function delete-subnetworks() {
  if [[ ${ENABLE_IP_ALIASES:-} != "true" ]]; then
    if [[ "${ENABLE_BIG_CLUSTER_SUBNETS}" = "true" ]]; then
      # If running in custom mode network we need to delete subnets
      mode="$(gcloud compute networks list ${NETWORK} --format='value(x_gcloud_mode)' || true)"
      if [[ "${mode}" == "custom" ]]; then
        echo "Deleting default subnets..."
        # This value should be kept in sync with number of regions.
        local parallelism=9
        gcloud compute networks subnets list --network="${NETWORK}" --format='value(region.basename())' | \
          xargs -i -P ${parallelism} gcloud --quiet compute networks subnets delete "${NETWORK}" --region="{}" || true
      fi
    fi
    return
  fi

  # Only delete automatically created subnets.
  if [[ ${IP_ALIAS_SUBNETWORK} == ${INSTANCE_PREFIX}-subnet-default ]]; then
    echo "Removing auto-created subnet ${NETWORK}:${IP_ALIAS_SUBNETWORK}"
    if [[ -n $(gcloud beta compute networks subnets describe \
          --project "${PROJECT}" \
          --region ${REGION} \
          ${IP_ALIAS_SUBNETWORK} 2>/dev/null) ]]; then
      gcloud beta --quiet compute networks subnets delete \
        --project "${PROJECT}" \
        --region ${REGION} \
        ${IP_ALIAS_SUBNETWORK}
    fi
  fi

  if [[ ${SERVICE_CLUSTER_IP_SUBNETWORK} == ${INSTANCE_PREFIX}-subnet-services ]]; then
    echo "Removing auto-created subnet ${NETWORK}:${SERVICE_CLUSTER_IP_SUBNETWORK}"
    if [[ -n $(gcloud beta compute networks subnets describe \
          --project "${PROJECT}" \
          --region ${REGION} \
          ${SERVICE_CLUSTER_IP_SUBNETWORK} 2>/dev/null) ]]; then
      gcloud --quiet beta compute networks subnets delete \
        --project "${PROJECT}" \
        --region ${REGION} \
        ${SERVICE_CLUSTER_IP_SUBNETWORK}
    fi
  fi
}

# Assumes:
#   NUM_NODES
# Sets:
#   MASTER_ROOT_DISK_SIZE
function get-master-root-disk-size() {
  if [[ "${NUM_NODES}" -le "1000" ]]; then
    export MASTER_ROOT_DISK_SIZE="20"
  else
    export MASTER_ROOT_DISK_SIZE="50"
  fi
}

# Assumes:
#   NUM_NODES
# Sets:
#   MASTER_DISK_SIZE
function get-master-disk-size() {
  if [[ "${NUM_NODES}" -le "1000" ]]; then
    export MASTER_DISK_SIZE="20GB"
  else
    export MASTER_DISK_SIZE="100GB"
  fi
}


# Generates SSL certificates for etcd cluster. Uses cfssl program.
#
# Assumed vars:
#   KUBE_TEMP: temporary directory
#
# Args:
#  $1: host name
#  $2: CA certificate
#  $3: CA key
#
# If CA cert/key is empty, the function will also generate certs for CA.
#
# Vars set:
#   ETCD_CA_KEY_BASE64
#   ETCD_CA_CERT_BASE64
#   ETCD_PEER_KEY_BASE64
#   ETCD_PEER_CERT_BASE64
#
function create-etcd-certs {
  local host=${1}
  local ca_cert=${2:-}
  local ca_key=${3:-}

  GEN_ETCD_CA_CERT="${ca_cert}" GEN_ETCD_CA_KEY="${ca_key}" \
    generate-etcd-cert "${KUBE_TEMP}/cfssl" "${host}" "peer" "peer"

  pushd "${KUBE_TEMP}/cfssl"
  ETCD_CA_KEY_BASE64=$(cat "ca-key.pem" | base64 | tr -d '\r\n')
  ETCD_CA_CERT_BASE64=$(cat "ca.pem" | gzip | base64 | tr -d '\r\n')
  ETCD_PEER_KEY_BASE64=$(cat "peer-key.pem" | base64 | tr -d '\r\n')
  ETCD_PEER_CERT_BASE64=$(cat "peer.pem" | gzip | base64 | tr -d '\r\n')
  popd
}

function create-master() {
  echo "Starting master and configuring firewalls"
  gcloud compute firewall-rules create "${MASTER_NAME}-https" \
    --project "${PROJECT}" \
    --network "${NETWORK}" \
    --target-tags "${MASTER_TAG}" \
    --allow tcp:443 &

  # We have to make sure the disk is created before creating the master VM, so
  # run this in the foreground.
  get-master-disk-size
  gcloud compute disks create "${MASTER_NAME}-pd" \
    --project "${PROJECT}" \
    --zone "${ZONE}" \
    --type "${MASTER_DISK_TYPE}" \
    --size "${MASTER_DISK_SIZE}"

  # Create disk for cluster registry if enabled
  if [[ "${ENABLE_CLUSTER_REGISTRY}" == true && -n "${CLUSTER_REGISTRY_DISK}" ]]; then
    gcloud compute disks create "${CLUSTER_REGISTRY_DISK}" \
      --project "${PROJECT}" \
      --zone "${ZONE}" \
      --type "${CLUSTER_REGISTRY_DISK_TYPE_GCE}" \
      --size "${CLUSTER_REGISTRY_DISK_SIZE}" &
  fi

  # Create rule for accessing and securing etcd servers.
  if ! gcloud compute firewall-rules --project "${PROJECT}" describe "${MASTER_NAME}-etcd" &>/dev/null; then
    gcloud compute firewall-rules create "${MASTER_NAME}-etcd" \
      --project "${PROJECT}" \
      --network "${NETWORK}" \
      --source-tags "${MASTER_TAG}" \
      --allow "tcp:2380,tcp:2381" \
      --target-tags "${MASTER_TAG}" &
  fi

  # Generate a bearer token for this cluster. We push this separately
  # from the other cluster variables so that the client (this
  # computer) can forget it later. This should disappear with
  # http://issue.k8s.io/3168
  KUBELET_TOKEN=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
  KUBE_PROXY_TOKEN=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
  if [[ "${ENABLE_NODE_PROBLEM_DETECTOR:-}" == "standalone" ]]; then
    NODE_PROBLEM_DETECTOR_TOKEN=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
  fi

  # Reserve the master's IP so that it can later be transferred to another VM
  # without disrupting the kubelets.
  create-static-ip "${MASTER_NAME}-ip" "${REGION}"
  MASTER_RESERVED_IP=$(gcloud compute addresses describe "${MASTER_NAME}-ip" \
    --project "${PROJECT}" --region "${REGION}" -q --format='value(address)')

  if [[ "${REGISTER_MASTER_KUBELET:-}" == "true" ]]; then
    KUBELET_APISERVER="${MASTER_RESERVED_IP}"
  fi

  KUBERNETES_MASTER_NAME="${MASTER_RESERVED_IP}"
  MASTER_ADVERTISE_ADDRESS="${MASTER_RESERVED_IP}"

  create-certs "${MASTER_RESERVED_IP}"
  create-etcd-certs ${MASTER_NAME}

  # Sets MASTER_ROOT_DISK_SIZE that is used by create-master-instance
  get-master-root-disk-size

  create-master-instance "${MASTER_RESERVED_IP}" &
}

# Adds master replica to etcd cluster.
#
# Assumed vars:
#   REPLICA_NAME
#   PROJECT
#   EXISTING_MASTER_NAME
#   EXISTING_MASTER_ZONE
#
# $1: etcd client port
# $2: etcd internal port
# returns the result of ssh command which adds replica
function add-replica-to-etcd() {
  local -r client_port="${1}"
  local -r internal_port="${2}"
  gcloud compute ssh "${EXISTING_MASTER_NAME}" \
    --project "${PROJECT}" \
    --zone "${EXISTING_MASTER_ZONE}" \
    --command \
      "curl localhost:${client_port}/v2/members -XPOST -H \"Content-Type: application/json\" -d '{\"peerURLs\":[\"https://${REPLICA_NAME}:${internal_port}\"]}' -s"
  return $?
}

# Sets EXISTING_MASTER_NAME and EXISTING_MASTER_ZONE variables.
#
# Assumed vars:
#   PROJECT
#
# NOTE: Must be in sync with get-replica-name-regexp
function set-existing-master() {
  local existing_master=$(gcloud compute instances list \
    --project "${PROJECT}" \
    --regexp "$(get-replica-name-regexp)" \
    --format "value(name,zone)" | head -n1)
  EXISTING_MASTER_NAME="$(echo "${existing_master}" | cut -f1)"
  EXISTING_MASTER_ZONE="$(echo "${existing_master}" | cut -f2)"
}

function replicate-master() {
  set-replica-name
  set-existing-master

  echo "Experimental: replicating existing master ${EXISTING_MASTER_ZONE}/${EXISTING_MASTER_NAME} as ${ZONE}/${REPLICA_NAME}"

  # Before we do anything else, we should configure etcd to expect more replicas.
  if ! add-replica-to-etcd 2379 2380; then
    echo "Failed to add master replica to etcd cluster."
    return 1
  fi
  if ! add-replica-to-etcd 4002 2381; then
    echo "Failed to add master replica to etcd events cluster."
    return 1
  fi

  # We have to make sure the disk is created before creating the master VM, so
  # run this in the foreground.
  get-master-disk-size
  gcloud compute disks create "${REPLICA_NAME}-pd" \
    --project "${PROJECT}" \
    --zone "${ZONE}" \
    --type "${MASTER_DISK_TYPE}" \
    --size "${MASTER_DISK_SIZE}"

  # Sets MASTER_ROOT_DISK_SIZE that is used by create-master-instance
  get-master-root-disk-size

  local existing_master_replicas="$(get-all-replica-names)"
  replicate-master-instance "${EXISTING_MASTER_ZONE}" "${EXISTING_MASTER_NAME}" "${existing_master_replicas}"

  # Add new replica to the load balancer.
  gcloud compute target-pools add-instances "${MASTER_NAME}" \
    --project "${PROJECT}" \
    --zone "${ZONE}" \
    --instances "${REPLICA_NAME}"
}

# Detaches old and ataches new external IP to a VM.
#
# Arguments:
#   $1 - VM name
#   $2 - VM zone
#   $3 - external static IP; if empty will use an ephemeral IP address.
function attach-external-ip() {
  local NAME=${1}
  local ZONE=${2}
  local IP_ADDR=${3:-}
  local ACCESS_CONFIG_NAME=$(gcloud compute instances describe "${NAME}" \
    --project "${PROJECT}" --zone "${ZONE}" \
    --format="value(networkInterfaces[0].accessConfigs[0].name)")
  gcloud compute instances delete-access-config "${NAME}" \
    --project "${PROJECT}" --zone "${ZONE}" \
    --access-config-name "${ACCESS_CONFIG_NAME}"
  if [[ -z ${IP_ADDR} ]]; then
    gcloud compute instances add-access-config "${NAME}" \
      --project "${PROJECT}" --zone "${ZONE}" \
      --access-config-name "${ACCESS_CONFIG_NAME}"
  else
    gcloud compute instances add-access-config "${NAME}" \
      --project "${PROJECT}" --zone "${ZONE}" \
      --access-config-name "${ACCESS_CONFIG_NAME}" \
      --address "${IP_ADDR}"
  fi
}

# Creates load balancer in front of apiserver if it doesn't exists already. Assumes there's only one
# existing master replica.
#
# Assumes:
#   PROJECT
#   MASTER_NAME
#   ZONE
#   REGION
function create-loadbalancer() {
  detect-master

  # Step 0: Return early if LB is already configured.
  if gcloud compute forwarding-rules describe ${MASTER_NAME} \
    --project "${PROJECT}" --region ${REGION} > /dev/null 2>&1; then
    echo "Load balancer already exists"
    return
  fi

  local EXISTING_MASTER_NAME="$(get-all-replica-names)"
  local EXISTING_MASTER_ZONE=$(gcloud compute instances list "${EXISTING_MASTER_NAME}" \
    --project "${PROJECT}" --format="value(zone)")

  echo "Creating load balancer in front of an already existing master in ${EXISTING_MASTER_ZONE}"

  # Step 1: Detach master IP address and attach ephemeral address to the existing master
  attach-external-ip "${EXISTING_MASTER_NAME}" "${EXISTING_MASTER_ZONE}"

  # Step 2: Create target pool.
  gcloud compute target-pools create "${MASTER_NAME}" --project "${PROJECT}" --region "${REGION}"
  # TODO: We should also add master instances with suffixes
  gcloud compute target-pools add-instances "${MASTER_NAME}" --instances "${EXISTING_MASTER_NAME}" --project "${PROJECT}" --zone "${EXISTING_MASTER_ZONE}"

  # Step 3: Create forwarding rule.
  # TODO: This step can take up to 20 min. We need to speed this up...
  gcloud compute forwarding-rules create ${MASTER_NAME} \
    --project "${PROJECT}" --region ${REGION} \
    --target-pool ${MASTER_NAME} --address=${KUBE_MASTER_IP} --ports=443

  echo -n "Waiting for the load balancer configuration to propagate..."
  local counter=0
  until $(curl -k -m1 https://${KUBE_MASTER_IP} &> /dev/null); do
    counter=$((counter+1))
    echo -n .
    if [[ ${counter} -ge 1800 ]]; then
      echo -e "${color_red}TIMEOUT${color_norm}" >&2
      echo -e "${color_red}Load balancer failed to initialize within ${counter} seconds.${color_norm}" >&2
      exit 2
    fi
  done
  echo "DONE"
}

function create-nodes-firewall() {
  # Create a single firewall rule for all minions.
  create-firewall-rule "${NODE_TAG}-all" "${CLUSTER_IP_RANGE}" "${NODE_TAG}" &

  # Report logging choice (if any).
  if [[ "${ENABLE_NODE_LOGGING-}" == "true" ]]; then
    echo "+++ Logging using Fluentd to ${LOGGING_DESTINATION:-unknown}"
  fi

  # Wait for last batch of jobs
  kube::util::wait-for-jobs || {
    echo -e "${color_red}Some commands failed.${color_norm}" >&2
  }
}

function create-nodes-template() {
  echo "Creating minions."

  # TODO(zmerlynn): Refactor setting scope flags.
  local scope_flags=
  if [[ -n "${NODE_SCOPES}" ]]; then
    scope_flags="--scopes ${NODE_SCOPES}"
  else
    scope_flags="--no-scopes"
  fi

  write-node-env

  local template_name="${NODE_INSTANCE_PREFIX}-template"

  create-node-instance-template $template_name
}

# Assumes:
# - MAX_INSTANCES_PER_MIG
# - NUM_NODES
# exports:
# - NUM_MIGS
function set_num_migs() {
  local defaulted_max_instances_per_mig=${MAX_INSTANCES_PER_MIG:-1000}

  if [[ ${defaulted_max_instances_per_mig} -le "0" ]]; then
    echo "MAX_INSTANCES_PER_MIG cannot be negative. Assuming default 1000"
    defaulted_max_instances_per_mig=1000
  fi
  export NUM_MIGS=$(((${NUM_NODES} + ${defaulted_max_instances_per_mig} - 1) / ${defaulted_max_instances_per_mig}))
}

# Assumes:
# - NUM_MIGS
# - NODE_INSTANCE_PREFIX
# - NUM_NODES
# - PROJECT
# - ZONE
function create-nodes() {
  local template_name="${NODE_INSTANCE_PREFIX}-template"

  local instances_left=${NUM_NODES}

  #TODO: parallelize this loop to speed up the process
  for ((i=1; i<=${NUM_MIGS}; i++)); do
    local group_name="${NODE_INSTANCE_PREFIX}-group-$i"
    if [[ $i == ${NUM_MIGS} ]]; then
      # TODO: We don't add a suffix for the last group to keep backward compatibility when there's only one MIG.
      # We should change it at some point, but note #18545 when changing this.
      group_name="${NODE_INSTANCE_PREFIX}-group"
    fi
    # Spread the remaining number of nodes evenly
    this_mig_size=$((${instances_left} / (${NUM_MIGS}-${i}+1)))
    instances_left=$((instances_left-${this_mig_size}))

    gcloud compute instance-groups managed \
        create "${group_name}" \
        --project "${PROJECT}" \
        --zone "${ZONE}" \
        --base-instance-name "${group_name}" \
        --size "${this_mig_size}" \
        --template "$template_name" || true;
    gcloud compute instance-groups managed wait-until-stable \
        "${group_name}" \
        --zone "${ZONE}" \
        --project "${PROJECT}" || true;
  done
}

# Assumes:
# - NUM_MIGS
# - NODE_INSTANCE_PREFIX
# - PROJECT
# - ZONE
# - AUTOSCALER_MAX_NODES
# - AUTOSCALER_MIN_NODES
# Exports
# - AUTOSCALER_MIG_CONFIG
function create-cluster-autoscaler-mig-config() {

  # Each MIG must have at least one node, so the min number of nodes
  # must be greater or equal to the number of migs.
  if [[ ${AUTOSCALER_MIN_NODES} -lt 0 ]]; then
    echo "AUTOSCALER_MIN_NODES must be greater or equal 0"
    exit 2
  fi

  # Each MIG must have at least one node, so the min number of nodes
  # must be greater or equal to the number of migs.
  if [[ ${AUTOSCALER_MAX_NODES} -lt ${NUM_MIGS} ]]; then
    echo "AUTOSCALER_MAX_NODES must be greater or equal ${NUM_MIGS}"
    exit 2
  fi

  # The code assumes that the migs were created with create-nodes
  # function which tries to evenly spread nodes across the migs.
  AUTOSCALER_MIG_CONFIG=""

  local left_min=${AUTOSCALER_MIN_NODES}
  local left_max=${AUTOSCALER_MAX_NODES}

  for ((i=1; i<=${NUM_MIGS}; i++)); do
    local group_name="${NODE_INSTANCE_PREFIX}-group-$i"
    if [[ $i == ${NUM_MIGS} ]]; then
      # TODO: We don't add a suffix for the last group to keep backward compatibility when there's only one MIG.
      # We should change it at some point, but note #18545 when changing this.
      group_name="${NODE_INSTANCE_PREFIX}-group"
    fi

    this_mig_min=$((${left_min}/(${NUM_MIGS}-${i}+1)))
    this_mig_max=$((${left_max}/(${NUM_MIGS}-${i}+1)))
    left_min=$((left_min-$this_mig_min))
    left_max=$((left_max-$this_mig_max))

    local mig_url="https://www.googleapis.com/compute/v1/projects/${PROJECT}/zones/${ZONE}/instanceGroups/${group_name}"
    AUTOSCALER_MIG_CONFIG="${AUTOSCALER_MIG_CONFIG} --nodes=${this_mig_min}:${this_mig_max}:${mig_url}"
  done

  AUTOSCALER_MIG_CONFIG="${AUTOSCALER_MIG_CONFIG} --scale-down-enabled=${AUTOSCALER_ENABLE_SCALE_DOWN}"
}

# Assumes:
# - NUM_MIGS
# - NODE_INSTANCE_PREFIX
# - PROJECT
# - ZONE
# - ENABLE_CLUSTER_AUTOSCALER
# - AUTOSCALER_MAX_NODES
# - AUTOSCALER_MIN_NODES
function create-autoscaler-config() {
  # Create autoscaler for nodes configuration if requested
  if [[ "${ENABLE_CLUSTER_AUTOSCALER}" == "true" ]]; then
    create-cluster-autoscaler-mig-config
    echo "Using autoscaler config: ${AUTOSCALER_MIG_CONFIG} ${AUTOSCALER_EXPANDER_CONFIG}"
  fi
}

function check-cluster() {
  detect-node-names
  detect-master

  echo "Waiting up to ${KUBE_CLUSTER_INITIALIZATION_TIMEOUT} seconds for cluster initialization."
  echo
  echo "  This will continually check to see if the API for kubernetes is reachable."
  echo "  This may time out if there was some uncaught error during start up."
  echo

  # curl in mavericks is borked.
  secure=""
  if which sw_vers >& /dev/null; then
    if [[ $(sw_vers | grep ProductVersion | awk '{print $2}') = "10.9."* ]]; then
      secure="--insecure"
    fi
  fi

  local start_time=$(date +%s)
  until curl --cacert "${CERT_DIR}/pki/ca.crt" \
          -H "Authorization: Bearer ${KUBE_BEARER_TOKEN}" \
          ${secure} \
          --max-time 5 --fail --output /dev/null --silent \
          "https://${KUBE_MASTER_IP}/api/v1/pods"; do
      local elapsed=$(($(date +%s) - ${start_time}))
      if [[ ${elapsed} -gt ${KUBE_CLUSTER_INITIALIZATION_TIMEOUT} ]]; then
          echo -e "${color_red}Cluster failed to initialize within ${KUBE_CLUSTER_INITIALIZATION_TIMEOUT} seconds.${color_norm}" >&2
          exit 2
      fi
      printf "."
      sleep 2
  done

  echo "Kubernetes cluster created."

  export KUBE_CERT="${CERT_DIR}/pki/issued/kubecfg.crt"
  export KUBE_KEY="${CERT_DIR}/pki/private/kubecfg.key"
  export CA_CERT="${CERT_DIR}/pki/ca.crt"
  export CONTEXT="${PROJECT}_${INSTANCE_PREFIX}"
  (
   umask 077

   # Update the user's kubeconfig to include credentials for this apiserver.
   create-kubeconfig

   create-kubeconfig-for-federation
  )

  # ensures KUBECONFIG is set
  get-kubeconfig-basicauth
  echo
  echo -e "${color_green}Kubernetes cluster is running.  The master is running at:"
  echo
  echo -e "${color_yellow}  https://${KUBE_MASTER_IP}"
  echo
  echo -e "${color_green}The user name and password to use is located in ${KUBECONFIG}.${color_norm}"
  echo

}

# Removes master replica from etcd cluster.
#
# Assumed vars:
#   REPLICA_NAME
#   PROJECT
#   EXISTING_MASTER_NAME
#   EXISTING_MASTER_ZONE
#
# $1: etcd client port
# returns the result of ssh command which removes replica
function remove-replica-from-etcd() {
  local -r port="${1}"
  [[ -n "${EXISTING_MASTER_NAME}" ]] || return
  gcloud compute ssh "${EXISTING_MASTER_NAME}" \
    --project "${PROJECT}" \
    --zone "${EXISTING_MASTER_ZONE}" \
    --command \
    "curl -s localhost:${port}/v2/members/\$(curl -s localhost:${port}/v2/members -XGET | sed 's/{\\\"id/\n/g' | grep ${REPLICA_NAME}\\\" | cut -f 3 -d \\\") -XDELETE -L 2>/dev/null"
  local -r res=$?
  echo "Removing etcd replica, name: ${REPLICA_NAME}, port: ${port}, result: ${res}"
  return "${res}"
}

# Delete a kubernetes cluster. This is called from test-teardown.
#
# Assumed vars:
#   MASTER_NAME
#   NODE_INSTANCE_PREFIX
#   ZONE
# This function tears down cluster resources 10 at a time to avoid issuing too many
# API calls and exceeding API quota. It is important to bring down the instances before bringing
# down the firewall rules and routes.
function kube-down() {
  local -r batch=200

  detect-project
  detect-node-names # For INSTANCE_GROUPS

  echo "Bringing down cluster"
  set +e  # Do not stop on error

  if [[ "${KUBE_DELETE_NODES:-}" != "false" ]]; then
    # Get the name of the managed instance group template before we delete the
    # managed instance group. (The name of the managed instance group template may
    # change during a cluster upgrade.)
    local templates=$(get-template "${PROJECT}")

    for group in ${INSTANCE_GROUPS[@]:-}; do
      if gcloud compute instance-groups managed describe "${group}" --project "${PROJECT}" --zone "${ZONE}" &>/dev/null; then
        gcloud compute instance-groups managed delete \
          --project "${PROJECT}" \
          --quiet \
          --zone "${ZONE}" \
          "${group}" &
      fi
    done

    # Wait for last batch of jobs
    kube::util::wait-for-jobs || {
      echo -e "Failed to delete instance group(s)." >&2
    }

    for template in ${templates[@]:-}; do
      if gcloud compute instance-templates describe --project "${PROJECT}" "${template}" &>/dev/null; then
        gcloud compute instance-templates delete \
          --project "${PROJECT}" \
          --quiet \
          "${template}"
      fi
    done
  fi

  local -r REPLICA_NAME="${KUBE_REPLICA_NAME:-$(get-replica-name)}"

  set-existing-master

  # Un-register the master replica from etcd and events etcd.
  remove-replica-from-etcd 2379
  remove-replica-from-etcd 4002

  # Delete the master replica (if it exists).
  if gcloud compute instances describe "${REPLICA_NAME}" --zone "${ZONE}" --project "${PROJECT}" &>/dev/null; then
    # If there is a load balancer in front of apiservers we need to first update its configuration.
    if gcloud compute target-pools describe "${MASTER_NAME}" --region "${REGION}" --project "${PROJECT}" &>/dev/null; then
      gcloud compute target-pools remove-instances "${MASTER_NAME}" \
        --project "${PROJECT}" \
        --zone "${ZONE}" \
        --instances "${REPLICA_NAME}"
    fi
    # Now we can safely delete the VM.
    gcloud compute instances delete \
      --project "${PROJECT}" \
      --quiet \
      --delete-disks all \
      --zone "${ZONE}" \
      "${REPLICA_NAME}"
  fi

  # Delete the master replica pd (possibly leaked by kube-up if master create failed).
  # TODO(jszczepkowski): remove also possibly leaked replicas' pds
  local -r replica_pd="${REPLICA_NAME:-${MASTER_NAME}}-pd"
  if gcloud compute disks describe "${replica_pd}" --zone "${ZONE}" --project "${PROJECT}" &>/dev/null; then
    gcloud compute disks delete \
      --project "${PROJECT}" \
      --quiet \
      --zone "${ZONE}" \
      "${replica_pd}"
  fi

  # Delete disk for cluster registry if enabled
  if [[ "${ENABLE_CLUSTER_REGISTRY}" == true && -n "${CLUSTER_REGISTRY_DISK}" ]]; then
    if gcloud compute disks describe "${CLUSTER_REGISTRY_DISK}" --zone "${ZONE}" --project "${PROJECT}" &>/dev/null; then
      gcloud compute disks delete \
        --project "${PROJECT}" \
        --quiet \
        --zone "${ZONE}" \
        "${CLUSTER_REGISTRY_DISK}"
    fi
  fi

  # Check if this are any remaining master replicas.
  local REMAINING_MASTER_COUNT=$(gcloud compute instances list \
    --project "${PROJECT}" \
    --regexp "$(get-replica-name-regexp)" \
    --format "value(zone)" | wc -l)

  # In the replicated scenario, if there's only a single master left, we should also delete load balancer in front of it.
  if [[ "${REMAINING_MASTER_COUNT}" -eq 1 ]]; then
    if gcloud compute forwarding-rules describe "${MASTER_NAME}" --region "${REGION}" --project "${PROJECT}" &>/dev/null; then
      detect-master
      local REMAINING_REPLICA_NAME="$(get-all-replica-names)"
      local REMAINING_REPLICA_ZONE=$(gcloud compute instances list "${REMAINING_REPLICA_NAME}" \
        --project "${PROJECT}" --format="value(zone)")
      gcloud compute forwarding-rules delete \
        --project "${PROJECT}" \
        --region "${REGION}" \
        --quiet \
        "${MASTER_NAME}"
      attach-external-ip "${REMAINING_REPLICA_NAME}" "${REMAINING_REPLICA_ZONE}" "${KUBE_MASTER_IP}"
      gcloud compute target-pools delete \
        --project "${PROJECT}" \
        --region "${REGION}" \
        --quiet \
        "${MASTER_NAME}"
    fi
  fi

  # If there are no more remaining master replicas, we should delete all remaining network resources.
  if [[ "${REMAINING_MASTER_COUNT}" -eq 0 ]]; then
    # Delete firewall rule for the master, etcd servers, and nodes.
    delete-firewall-rules "${MASTER_NAME}-https" "${MASTER_NAME}-etcd" "${NODE_TAG}-all"
    # Delete the master's reserved IP
    if gcloud compute addresses describe "${MASTER_NAME}-ip" --region "${REGION}" --project "${PROJECT}" &>/dev/null; then
      gcloud compute addresses delete \
        --project "${PROJECT}" \
        --region "${REGION}" \
        --quiet \
        "${MASTER_NAME}-ip"
    fi
  fi

  if [[ "${KUBE_DELETE_NODES:-}" != "false" ]]; then
    # Find out what minions are running.
    local -a minions
    minions=( $(gcloud compute instances list \
                  --project "${PROJECT}" --zones "${ZONE}" \
                  --regexp "${NODE_INSTANCE_PREFIX}-.+" \
                  --format='value(name)') )
    # If any minions are running, delete them in batches.
    while (( "${#minions[@]}" > 0 )); do
      echo Deleting nodes "${minions[*]::${batch}}"
      gcloud compute instances delete \
        --project "${PROJECT}" \
        --quiet \
        --delete-disks boot \
        --zone "${ZONE}" \
        "${minions[@]::${batch}}"
      minions=( "${minions[@]:${batch}}" )
    done
  fi

  # If there are no more remaining master replicas: delete routes, pd for influxdb and update kubeconfig
  if [[ "${REMAINING_MASTER_COUNT}" -eq 0 ]]; then
    # Delete routes.
    local -a routes
    # Clean up all routes w/ names like "<cluster-name>-<node-GUID>"
    # e.g. "kubernetes-12345678-90ab-cdef-1234-567890abcdef". The name is
    # determined by the node controller on the master.
    # Note that this is currently a noop, as synchronously deleting the node MIG
    # first allows the master to cleanup routes itself.
    local TRUNCATED_PREFIX="${INSTANCE_PREFIX:0:26}"
    routes=( $(gcloud compute routes list --project "${PROJECT}" \
      --regexp "${TRUNCATED_PREFIX}-.{8}-.{4}-.{4}-.{4}-.{12}"  \
      --format='value(name)') )
    while (( "${#routes[@]}" > 0 )); do
      echo Deleting routes "${routes[*]::${batch}}"
      gcloud compute routes delete \
        --project "${PROJECT}" \
        --quiet \
        "${routes[@]::${batch}}"
      routes=( "${routes[@]:${batch}}" )
    done

    # Delete persistent disk for influx-db.
    if gcloud compute disks describe "${INSTANCE_PREFIX}"-influxdb-pd --zone "${ZONE}" --project "${PROJECT}" &>/dev/null; then
      gcloud compute disks delete \
        --project "${PROJECT}" \
        --quiet \
        --zone "${ZONE}" \
        "${INSTANCE_PREFIX}"-influxdb-pd
    fi

    # Delete all remaining firewall rules and network.
    delete-firewall-rules \
      "${CLUSTER_NAME}-default-internal-master" \
      "${CLUSTER_NAME}-default-internal-node" \
      "${NETWORK}-default-ssh" \
      "${NETWORK}-default-internal"  # Pre-1.5 clusters

    if [[ "${KUBE_DELETE_NETWORK}" == "true" ]]; then
      delete-subnetworks || true
      delete-network || true  # might fail if there are leaked firewall rules
    fi

    # If there are no more remaining master replicas, we should update kubeconfig.
    export CONTEXT="${PROJECT}_${INSTANCE_PREFIX}"
    clear-kubeconfig
  else
  # If some master replicas remain: cluster has been changed, we need to re-validate it.
    echo "... calling validate-cluster" >&2
    # Override errexit
    (validate-cluster) && validate_result="$?" || validate_result="$?"

    # We have two different failure modes from validate cluster:
    # - 1: fatal error - cluster won't be working correctly
    # - 2: weak error - something went wrong, but cluster probably will be working correctly
    # We just print an error message in case 2).
    if [[ "${validate_result}" -eq 1 ]]; then
      exit 1
    elif [[ "${validate_result}" -eq 2 ]]; then
      echo "...ignoring non-fatal errors in validate-cluster" >&2
    fi
  fi
  set -e
}

# Prints name of one of the master replicas in the current zone. It will be either
# just MASTER_NAME or MASTER_NAME with a suffix for a replica (see get-replica-name-regexp).
#
# Assumed vars:
#   PROJECT
#   ZONE
#   MASTER_NAME
#
# NOTE: Must be in sync with get-replica-name-regexp and set-replica-name.
function get-replica-name() {
  echo $(gcloud compute instances list \
    --project "${PROJECT}" \
    --zones "${ZONE}" \
    --regexp "$(get-replica-name-regexp)" \
    --format "value(name)" | head -n1)
}

# Prints comma-separated names of all of the master replicas in all zones.
#
# Assumed vars:
#   PROJECT
#   MASTER_NAME
#
# NOTE: Must be in sync with get-replica-name-regexp and set-replica-name.
function get-all-replica-names() {
  echo $(gcloud compute instances list \
    --project "${PROJECT}" \
    --regexp "$(get-replica-name-regexp)" \
    --format "value(name)" | tr "\n" "," | sed 's/,$//')
}

# Prints the number of all of the master replicas in all zones.
#
# Assumed vars:
#   MASTER_NAME
function get-master-replicas-count() {
  detect-project
  local num_masters=$(gcloud compute instances list \
    --project "${PROJECT}" \
    --regexp "$(get-replica-name-regexp)" \
    --format "value(zone)" | wc -l)
  echo -n "${num_masters}"
}

# Prints regexp for full master machine name. In a cluster with replicated master,
# VM names may either be MASTER_NAME or MASTER_NAME with a suffix for a replica.
function get-replica-name-regexp() {
  echo "${MASTER_NAME}(-...)?"
}

# Sets REPLICA_NAME to a unique name for a master replica that will match
# expected regexp (see get-replica-name-regexp).
#
# Assumed vars:
#   PROJECT
#   ZONE
#   MASTER_NAME
#
# Sets:
#   REPLICA_NAME
function set-replica-name() {
  local instances=$(gcloud compute instances list \
    --project "${PROJECT}" \
    --regexp "$(get-replica-name-regexp)" \
    --format "value(name)")

  suffix=""
  while echo "${instances}" | grep "${suffix}" &>/dev/null; do
    suffix="$(date | md5sum | head -c3)"
  done
  REPLICA_NAME="${MASTER_NAME}-${suffix}"
}

# Gets the instance template for given NODE_INSTANCE_PREFIX. It echos the template name so that the function
# output can be used.
# Assumed vars:
#   NODE_INSTANCE_PREFIX
#
# $1: project
function get-template() {
  gcloud compute instance-templates list -r "${NODE_INSTANCE_PREFIX}-template(-(${KUBE_RELEASE_VERSION_DASHED_REGEX}|${KUBE_CI_VERSION_DASHED_REGEX}))?" \
    --project="${1}" --format='value(name)'
}

# Checks if there are any present resources related kubernetes cluster.
#
# Assumed vars:
#   MASTER_NAME
#   NODE_INSTANCE_PREFIX
#   ZONE
#   REGION
# Vars set:
#   KUBE_RESOURCE_FOUND
function check-resources() {
  detect-project
  detect-node-names

  echo "Looking for already existing resources"
  KUBE_RESOURCE_FOUND=""

  if [[ -n "${INSTANCE_GROUPS[@]:-}" ]]; then
    KUBE_RESOURCE_FOUND="Managed instance groups ${INSTANCE_GROUPS[@]}"
    return 1
  fi

  if gcloud compute instance-templates describe --project "${PROJECT}" "${NODE_INSTANCE_PREFIX}-template" &>/dev/null; then
    KUBE_RESOURCE_FOUND="Instance template ${NODE_INSTANCE_PREFIX}-template"
    return 1
  fi

  if gcloud compute instances describe --project "${PROJECT}" "${MASTER_NAME}" --zone "${ZONE}" &>/dev/null; then
    KUBE_RESOURCE_FOUND="Kubernetes master ${MASTER_NAME}"
    return 1
  fi

  if gcloud compute disks describe --project "${PROJECT}" "${MASTER_NAME}"-pd --zone "${ZONE}" &>/dev/null; then
    KUBE_RESOURCE_FOUND="Persistent disk ${MASTER_NAME}-pd"
    return 1
  fi

  if gcloud compute disks describe --project "${PROJECT}" "${CLUSTER_REGISTRY_DISK}" --zone "${ZONE}" &>/dev/null; then
    KUBE_RESOURCE_FOUND="Persistent disk ${CLUSTER_REGISTRY_DISK}"
    return 1
  fi

  # Find out what minions are running.
  local -a minions
  minions=( $(gcloud compute instances list \
                --project "${PROJECT}" --zones "${ZONE}" \
                --regexp "${NODE_INSTANCE_PREFIX}-.+" \
                --format='value(name)') )
  if (( "${#minions[@]}" > 0 )); then
    KUBE_RESOURCE_FOUND="${#minions[@]} matching matching ${NODE_INSTANCE_PREFIX}-.+"
    return 1
  fi

  if gcloud compute firewall-rules describe --project "${PROJECT}" "${MASTER_NAME}-https" &>/dev/null; then
    KUBE_RESOURCE_FOUND="Firewall rules for ${MASTER_NAME}-https"
    return 1
  fi

  if gcloud compute firewall-rules describe --project "${PROJECT}" "${NODE_TAG}-all" &>/dev/null; then
    KUBE_RESOURCE_FOUND="Firewall rules for ${MASTER_NAME}-all"
    return 1
  fi

  local -a routes
  routes=( $(gcloud compute routes list --project "${PROJECT}" \
    --regexp "${INSTANCE_PREFIX}-minion-.{4}" --format='value(name)') )
  if (( "${#routes[@]}" > 0 )); then
    KUBE_RESOURCE_FOUND="${#routes[@]} routes matching ${INSTANCE_PREFIX}-minion-.{4}"
    return 1
  fi

  if gcloud compute addresses describe --project "${PROJECT}" "${MASTER_NAME}-ip" --region "${REGION}" &>/dev/null; then
    KUBE_RESOURCE_FOUND="Master's reserved IP"
    return 1
  fi

  # No resources found.
  return 0
}

# Prepare to push new binaries to kubernetes cluster
#  $1 - whether prepare push to node
function prepare-push() {
  local node="${1-}"
  #TODO(dawnchen): figure out how to upgrade a Container Linux node
  if [[ "${node}" == "true" && "${NODE_OS_DISTRIBUTION}" != "debian" ]]; then
    echo "Updating nodes in a kubernetes cluster with ${NODE_OS_DISTRIBUTION} is not supported yet." >&2
    exit 1
  fi
  if [[ "${node}" != "true" && "${MASTER_OS_DISTRIBUTION}" != "debian" ]]; then
    echo "Updating the master in a kubernetes cluster with ${MASTER_OS_DISTRIBUTION} is not supported yet." >&2
    exit 1
  fi

  OUTPUT=${KUBE_ROOT}/_output/logs
  mkdir -p ${OUTPUT}

  kube::util::ensure-temp-dir
  detect-project
  detect-master
  detect-node-names
  get-kubeconfig-basicauth
  get-kubeconfig-bearertoken

  # Make sure we have the tar files staged on Google Storage
  tars_from_version

  # Prepare node env vars and update MIG template
  if [[ "${node}" == "true" ]]; then
    write-node-env

    # TODO(zmerlynn): Refactor setting scope flags.
    local scope_flags=
    if [[ -n "${NODE_SCOPES}" ]]; then
      scope_flags="--scopes ${NODE_SCOPES}"
    else
      scope_flags="--no-scopes"
    fi

    # Ugly hack: Since it is not possible to delete instance-template that is currently
    # being used, create a temp one, then delete the old one and recreate it once again.
    local tmp_template_name="${NODE_INSTANCE_PREFIX}-template-tmp"
    create-node-instance-template $tmp_template_name

    local template_name="${NODE_INSTANCE_PREFIX}-template"
    for group in ${INSTANCE_GROUPS[@]:-}; do
      gcloud compute instance-groups managed \
        set-instance-template "${group}" \
        --template "$tmp_template_name" \
        --zone "${ZONE}" \
        --project "${PROJECT}" || true;
    done

    gcloud compute instance-templates delete \
      --project "${PROJECT}" \
      --quiet \
      "$template_name" || true

    create-node-instance-template "$template_name"

    for group in ${INSTANCE_GROUPS[@]:-}; do
      gcloud compute instance-groups managed \
        set-instance-template "${group}" \
        --template "$template_name" \
        --zone "${ZONE}" \
        --project "${PROJECT}" || true;
    done

    gcloud compute instance-templates delete \
      --project "${PROJECT}" \
      --quiet \
      "$tmp_template_name" || true
  fi
}

# Push binaries to kubernetes master
function push-master() {
  echo "Updating master metadata ..."
  write-master-env
  prepare-startup-script
  add-instance-metadata-from-file "${KUBE_MASTER}" "kube-env=${KUBE_TEMP}/master-kube-env.yaml" "startup-script=${KUBE_TEMP}/configure-vm.sh"

  echo "Pushing to master (log at ${OUTPUT}/push-${KUBE_MASTER}.log) ..."
  cat ${KUBE_TEMP}/configure-vm.sh | gcloud compute ssh --ssh-flag="-o LogLevel=quiet" --project "${PROJECT}" --zone "${ZONE}" "${KUBE_MASTER}" --command "sudo bash -s -- --push" &> ${OUTPUT}/push-"${KUBE_MASTER}".log
}

# Push binaries to kubernetes node
function push-node() {
  node=${1}

  echo "Updating node ${node} metadata... "
  prepare-startup-script
  add-instance-metadata-from-file "${node}" "kube-env=${KUBE_TEMP}/node-kube-env.yaml" "startup-script=${KUBE_TEMP}/configure-vm.sh"

  echo "Start upgrading node ${node} (log at ${OUTPUT}/push-${node}.log) ..."
  cat ${KUBE_TEMP}/configure-vm.sh | gcloud compute ssh --ssh-flag="-o LogLevel=quiet" --project "${PROJECT}" --zone "${ZONE}" "${node}" --command "sudo bash -s -- --push" &> ${OUTPUT}/push-"${node}".log
}

# Push binaries to kubernetes cluster
function kube-push() {
  # Disable this until it's fixed.
  # See https://github.com/kubernetes/kubernetes/issues/17397
  echo "./cluster/kube-push.sh is currently not supported in GCE."
  echo "Please use ./cluster/gce/upgrade.sh."
  exit 1

  prepare-push true

  push-master

  for (( i=0; i<${#NODE_NAMES[@]}; i++)); do
    push-node "${NODE_NAMES[$i]}" &
  done

  kube::util::wait-for-jobs || {
    echo -e "${color_red}Some commands failed.${color_norm}" >&2
  }

  # TODO(zmerlynn): Re-create instance-template with the new
  # node-kube-env. This isn't important until the node-ip-range issue
  # is solved (because that's blocking automatic dynamic nodes from
  # working). The node-kube-env has to be composed with the KUBELET_TOKEN
  # and KUBE_PROXY_TOKEN.  Ideally we would have
  # http://issue.k8s.io/3168
  # implemented before then, though, so avoiding this mess until then.

  echo
  echo "Kubernetes cluster is running.  The master is running at:"
  echo
  echo "  https://${KUBE_MASTER_IP}"
  echo
  echo "The user name and password to use is located in ~/.kube/config"
  echo
}

# -----------------------------------------------------------------------------
# Cluster specific test helpers used from hack/e2e.go

# Execute prior to running tests to build a release if required for env.
#
# Assumed Vars:
#   KUBE_ROOT
function test-build-release() {
  # Make a release
  "${KUBE_ROOT}/build/release.sh"
}

# Execute prior to running tests to initialize required structure. This is
# called from hack/e2e.go only when running -up.
#
# Assumed vars:
#   Variables from config.sh
function test-setup() {
  # Detect the project into $PROJECT if it isn't set
  detect-project

  if [[ ${MULTIZONE:-} == "true" && -n ${E2E_ZONES:-} ]]; then
    for KUBE_GCE_ZONE in ${E2E_ZONES}
    do
      KUBE_GCE_ZONE="${KUBE_GCE_ZONE}" KUBE_USE_EXISTING_MASTER="${KUBE_USE_EXISTING_MASTER:-}" "${KUBE_ROOT}/cluster/kube-up.sh"
      KUBE_USE_EXISTING_MASTER="true" # For subsequent zones we use the existing master
    done
  else
    "${KUBE_ROOT}/cluster/kube-up.sh"
  fi

  # Open up port 80 & 8080 so common containers on minions can be reached
  # TODO(roberthbailey): Remove this once we are no longer relying on hostPorts.
  local start=`date +%s`
  gcloud compute firewall-rules create \
    --project "${PROJECT}" \
    --target-tags "${NODE_TAG}" \
    --allow tcp:80,tcp:8080 \
    --network "${NETWORK}" \
    "${NODE_TAG}-${INSTANCE_PREFIX}-http-alt" 2> /dev/null || true
  # As there is no simple way to wait longer for this operation we need to manually
  # wait some additional time (20 minutes altogether).
  while ! gcloud compute firewall-rules describe --project "${PROJECT}" "${NODE_TAG}-${INSTANCE_PREFIX}-http-alt" 2> /dev/null; do
    if [[ $(($start + 1200)) -lt `date +%s` ]]; then
      echo -e "${color_red}Failed to create firewall ${NODE_TAG}-${INSTANCE_PREFIX}-http-alt in ${PROJECT}" >&2
      exit 1
    fi
    sleep 5
  done

  # Open up the NodePort range
  # TODO(justinsb): Move to main setup, if we decide whether we want to do this by default.
  start=`date +%s`
  gcloud compute firewall-rules create \
    --project "${PROJECT}" \
    --target-tags "${NODE_TAG}" \
    --allow tcp:30000-32767,udp:30000-32767 \
    --network "${NETWORK}" \
    "${NODE_TAG}-${INSTANCE_PREFIX}-nodeports" 2> /dev/null || true
  # As there is no simple way to wait longer for this operation we need to manually
  # wait some additional time (20 minutes altogether).
  while ! gcloud compute firewall-rules describe --project "${PROJECT}" "${NODE_TAG}-${INSTANCE_PREFIX}-nodeports" 2> /dev/null; do
    if [[ $(($start + 1200)) -lt `date +%s` ]]; then
      echo -e "${color_red}Failed to create firewall ${NODE_TAG}-${INSTANCE_PREFIX}-nodeports in ${PROJECT}" >&2
      exit 1
    fi
    sleep 5
  done
}

# Execute after running tests to perform any required clean-up. This is called
# from hack/e2e.go
function test-teardown() {
  detect-project
  echo "Shutting down test cluster in background."
  delete-firewall-rules \
    "${NODE_TAG}-${INSTANCE_PREFIX}-http-alt" \
    "${NODE_TAG}-${INSTANCE_PREFIX}-nodeports"
  if [[ ${MULTIZONE:-} == "true" && -n ${E2E_ZONES:-} ]]; then
      local zones=( ${E2E_ZONES} )
      # tear them down in reverse order, finally tearing down the master too.
      for ((zone_num=${#zones[@]}-1; zone_num>0; zone_num--))
      do
	  KUBE_GCE_ZONE="${zones[zone_num]}" KUBE_USE_EXISTING_MASTER="true" "${KUBE_ROOT}/cluster/kube-down.sh"
      done
      KUBE_GCE_ZONE="${zones[0]}" KUBE_USE_EXISTING_MASTER="false" "${KUBE_ROOT}/cluster/kube-down.sh"
  else
      "${KUBE_ROOT}/cluster/kube-down.sh"
  fi
}

# SSH to a node by name ($1) and run a command ($2).
function ssh-to-node() {
  local node="$1"
  local cmd="$2"
  # Loop until we can successfully ssh into the box
  for try in {1..5}; do
    if gcloud compute ssh --ssh-flag="-o LogLevel=quiet" --ssh-flag="-o ConnectTimeout=30" --project "${PROJECT}" --zone="${ZONE}" "${node}" --command "echo test > /dev/null"; then
      break
    fi
    sleep 5
  done
  # Then actually try the command.
  gcloud compute ssh --ssh-flag="-o LogLevel=quiet" --ssh-flag="-o ConnectTimeout=30" --project "${PROJECT}" --zone="${ZONE}" "${node}" --command "${cmd}"
}

# Perform preparations required to run e2e tests
function prepare-e2e() {
  detect-project
}

# Writes configure-vm.sh to a temporary location with comments stripped. GCE
# limits the size of metadata fields to 32K, and stripping comments is the
# easiest way to buy us a little more room.
function prepare-startup-script() {
  # Find a standard sed instance (and ensure that the command works as expected on a Mac).
  SED=sed
  if which gsed &>/dev/null; then
    SED=gsed
  fi
  if ! ($SED --version 2>&1 | grep -q GNU); then
    echo "!!! GNU sed is required.  If on OS X, use 'brew install gnu-sed'."
    exit 1
  fi
  $SED '/^\s*#\([^!].*\)*$/ d' ${KUBE_ROOT}/cluster/gce/configure-vm.sh > ${KUBE_TEMP}/configure-vm.sh
}
