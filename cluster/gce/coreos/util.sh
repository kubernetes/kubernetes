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

# A library of helper functions and constant for the local config.

# Use the config file specified in $KUBE_CONFIG_FILE, or default to
# config-default.sh.
KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../../..
source "${KUBE_ROOT}/cluster/gce/coreos/${KUBE_CONFIG_FILE-"config-default.sh"}"

NODE_INSTANCE_PREFIX="${INSTANCE_PREFIX}-minion"

# Verify prereqs
function verify-prereqs {
  local cmd
  for cmd in gcloud gsutil; do
    which "${cmd}" >/dev/null || {
      echo "Can't find ${cmd} in PATH, please fix and retry. The Google Cloud "
      echo "SDK can be downloaded from https://cloud.google.com/sdk/."
      exit 1
    }
  done
}

# Create a temp dir that'll be deleted at the end of this bash session.
#
# Vars set:
#   KUBE_TEMP
function ensure-temp-dir {
  if [[ -z ${KUBE_TEMP-} ]]; then
    KUBE_TEMP=$(mktemp -d -t kubernetes.XXXXXX)
    trap 'rm -rf "${KUBE_TEMP}"' EXIT
  fi
}

# Verify and find the various tar files that we are going to use on the server.
#
# Vars set:
#   SERVER_BINARY_TAR
function find-release-tars {
  SERVER_BINARY_TAR="${KUBE_ROOT}/server/kubernetes-server-linux-amd64.tar.gz"
  if [[ ! -f "$SERVER_BINARY_TAR" ]]; then
    SERVER_BINARY_TAR="${KUBE_ROOT}/_output/release-tars/kubernetes-server-linux-amd64.tar.gz"
  fi
  if [[ ! -f "$SERVER_BINARY_TAR" ]]; then
    echo "!!! Cannot find kubernetes-server-linux-amd64.tar.gz"
    exit 1
  fi
}

# Use the gcloud defaults to find the project.  If it is already set in the
# environment then go with that.
#
# Vars set:
#   PROJECT
#   PROJECT_REPORTED
function detect-project () {
  if [[ -z "${PROJECT-}" ]]; then
    PROJECT=$(gcloud config list project | tail -n 1 | cut -f 3 -d ' ')
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


# Take the local tar files and upload them to Google Storage.  They will then be
# downloaded by the master as part of the start up script for the master.
#
# Assumed vars:
#   PROJECT
#   SERVER_BINARY_TAR
# Vars set:
#   SERVER_BINARY_TAR_URL
function upload-server-tars() {
  SERVER_BINARY_TAR_URL=

  local project_hash
  if which md5 > /dev/null 2>&1; then
    project_hash=$(md5 -q -s "$PROJECT")
  else
    project_hash=$(echo -n "$PROJECT" | md5sum | awk '{ print $1 }')
  fi
  project_hash=${project_hash:0:5}

  local -r staging_bucket="gs://kubernetes-staging-${project_hash}"

  # Ensure the bucket is created
  if ! gsutil ls "$staging_bucket" > /dev/null 2>&1 ; then
    echo "Creating $staging_bucket"
    gsutil mb "${staging_bucket}"
  fi

  local -r staging_path="${staging_bucket}/devel"

  echo "+++ Staging server tars to Google Storage: ${staging_path}"
  local server_binary_gs_url="${staging_path}/${SERVER_BINARY_TAR##*/}"
  gsutil -q -h "Cache-Control:private, max-age=0" cp "${SERVER_BINARY_TAR}" "${server_binary_gs_url}"
  gsutil acl ch -g all:R "${server_binary_gs_url}" >/dev/null 2>&1

  # Convert from gs:// URL to an https:// URL
  SERVER_BINARY_TAR_URL="${server_binary_gs_url/gs:\/\//https://storage.googleapis.com/}"
}

# Detect minions created in the minion group
#
# Assumed vars:
#   NODE_INSTANCE_PREFIX
# Vars set:
#   MINION_NAMES
function detect-minion-names {
  detect-project
  MINION_NAMES=($(gcloud preview --project "${PROJECT}" instance-groups \
    --zone "${ZONE}" instances --group "${NODE_INSTANCE_PREFIX}-group" list \
    | cut -d'/' -f11))
  echo "MINION_NAMES=${MINION_NAMES[*]}"
}

# Waits until the number of running nodes in the instance group is equal to NUM_NODES
#
# Assumed vars:
#   NODE_INSTANCE_PREFIX
#   NUM_MINIONS
function wait-for-minions-to-run {
  detect-project
  local running_minions=0
  while [[ "${NUM_MINIONS}" != "${running_minions}" ]]; do
    echo -e -n "${color_yellow}Waiting for minions to run. "
    echo -e "${running_minions} out of ${NUM_MINIONS} running. Retrying.${color_norm}"
    sleep 5
    running_minions=$(gcloud preview --project "${PROJECT}" instance-groups \
      --zone "${ZONE}" instances --group "${NODE_INSTANCE_PREFIX}-group" list \
      --running | wc -l | xargs)
  done
}

# Detect the information about the minions
#
# Assumed vars:
#   ZONE
# Vars set:
#   MINION_NAMES
#   KUBE_MINION_IP_ADDRESSES (array)
function detect-minions () {
  detect-project
  detect-minion-names
  KUBE_MINION_IP_ADDRESSES=()
  for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
    local minion_ip=$(gcloud compute instances describe --project "${PROJECT}" --zone "${ZONE}" \
      "${MINION_NAMES[$i]}" --fields networkInterfaces[0].accessConfigs[0].natIP \
      --format=text | awk '{ print $2 }')
    if [[ -z "${minion_ip-}" ]] ; then
      echo "Did not find ${MINION_NAMES[$i]}" >&2
    else
      echo "Found ${MINION_NAMES[$i]} at ${minion_ip}"
      KUBE_MINION_IP_ADDRESSES+=("${minion_ip}")
    fi
  done
  if [[ -z "${KUBE_MINION_IP_ADDRESSES-}" ]]; then
    echo "Could not detect Kubernetes minion nodes.  Make sure you've launched a cluster with 'kube-up.sh'" >&2
    exit 1
  fi
}

# Detect the IP for the master
#
# Assumed vars:
#   MASTER_NAME
#   ZONE
# Vars set:
#   KUBE_MASTER
#   KUBE_MASTER_IP
#   KUBE_MASTER_INTERNAL_IP
function detect-master () {
  detect-project
  KUBE_MASTER=${MASTER_NAME}
  if [[ -z "${KUBE_MASTER_IP-}" ]]; then
    KUBE_MASTER_IP=$(gcloud compute instances describe --project "${PROJECT}" --zone "${ZONE}" \
      "${MASTER_NAME}" --fields networkInterfaces[0].accessConfigs[0].natIP \
      --format=text | awk '{ print $2 }')
  fi
  if [[ -z "${KUBE_MASTER_IP-}" ]]; then
    echo "Could not detect Kubernetes master node (external IP).  Make sure you've launched a cluster with 'kube-up.sh'" >&2
    exit 1
  fi
  if [[ -z "${KUBE_MASTER_INTERNAL_IP-}" ]]; then
    KUBE_MASTER_INTERNAL_IP=$(gcloud compute instances describe --project "${PROJECT}" --zone "${ZONE}" \
      "${MASTER_NAME}" --fields networkInterfaces[0].networkIP \
      --format=text | awk '{ print $2 }')
  fi
  if [[ -z "${KUBE_MASTER_INTERNAL_IP-}" ]]; then
    echo "Could not detect Kubernetes master node (internal IP).  Make sure you've launched a cluster with 'kube-up.sh'" >&2
    exit 1
  fi
  echo "Using master: $KUBE_MASTER (external IP: $KUBE_MASTER_IP, internal IP: $KUBE_MASTER_INTERNAL_IP)"
}

# Ensure that we have a password created for validating to the master.  Will
# read from the kubernetes auth-file for the current context if available.
#
# Assumed vars
#   KUBE_ROOT
#
# Vars set:
#   KUBE_USER
#   KUBE_PASSWORD
function get-password {
  # go template to extract the auth-path of the current-context user
  # Note: we save dot ('.') to $dot because the 'with' action overrides dot
  local template='{{$dot := .}}{{with $ctx := index $dot "current-context"}}{{$user := index $dot "contexts" $ctx "user"}}{{index $dot "users" $user "auth-path"}}{{end}}'
  local file=$("${KUBE_ROOT}/cluster/kubectl.sh" config view -o template --template="${template}")
  if [[ ! -z "$file" && -r "$file" ]]; then
    KUBE_USER=$(cat "$file" | python -c 'import json,sys;print json.load(sys.stdin)["User"]')
    KUBE_PASSWORD=$(cat "$file" | python -c 'import json,sys;print json.load(sys.stdin)["Password"]')
    return
  fi
  KUBE_USER=admin
  KUBE_PASSWORD=$(python -c 'import string,random; print "".join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(16))')
}

# Generate authentication token for admin user. Will
# read from $HOME/.kubernetes_auth if available.
#
# Vars set:
#   KUBE_ADMIN_TOKEN
function get-admin-token {
  local file="$HOME/.kubernetes_auth"
  if [[ -r "$file" ]]; then
    KUBE_ADMIN_TOKEN=$(cat "$file" | python -c 'import json,sys;print json.load(sys.stdin)["BearerToken"]')
    return
  fi
  KUBE_ADMIN_TOKEN=$(python -c 'import string,random; print "".join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(32))')
}

# Wait for background jobs to finish. Exit with
# an error status if any of the jobs failed.
function wait-for-jobs {
  local fail=0
  local job
  for job in $(jobs -p); do
    wait "${job}" || fail=$((fail + 1))
  done
  if (( fail != 0 )); then
    echo -e "${color_red}${fail} commands failed.  Exiting.${color_norm}" >&2
    # Ignore failures for now.
    # exit 2
  fi
}

# Robustly try to create a firewall rule.
# $1: The name of firewall rule.
# $2: IP ranges.
# $3: Target tags for this firewall rule.
function create-firewall-rule {
  detect-project
  local attempt=0
  while true; do
    if ! gcloud compute firewall-rules create "$1" \
      --project "${PROJECT}" \
      --network "${NETWORK}" \
      --source-ranges "$2" \
      --target-tags "$3" \
      --allow tcp udp icmp esp ah sctp; then
        if (( attempt > 5 )); then
          echo -e "${color_red}Failed to create firewall rule $1 ${color_norm}"
          exit 2
        fi
        echo -e "${color_yellow}Attempt $(($attempt+1)) failed to create firewall rule $1. Retrying.${color_norm}"
        attempt=$(($attempt+1))
    else
        break
    fi
  done
}

# Robustly try to create a route.
# $1: The name of the route.
# $2: IP range.
function create-route {
  detect-project
  local attempt=0
  while true; do
    if ! gcloud compute routes create "$1" \
      --project "${PROJECT}" \
      --destination-range "$2" \
      --network "${NETWORK}" \
      --next-hop-instance "$1" \
      --next-hop-instance-zone "${ZONE}"; then
        if (( attempt > 5 )); then
          echo -e "${color_red}Failed to create route $1 ${color_norm}"
          exit 2
        fi
        echo -e "${color_yellow}Attempt $(($attempt+1)) failed to create route $1. Retrying.${color_norm}"
        attempt=$(($attempt+1))
    else
        break
    fi
  done
}

# Robustly try to create an instance template.
# $1: The name of the instance template.
# $2: The scopes flag.
# $3: The local path of container_bridge.sh file
function create-node-template {
  detect-project
  local attempt=0
  while true; do
    if ! gcloud compute instance-templates create "$1" \
      --project "${PROJECT}" \
      --machine-type "${MINION_SIZE}" \
      --boot-disk-type "${MINION_DISK_TYPE}" \
      --boot-disk-size "${MINION_DISK_SIZE}" \
      --image-project="${IMAGE_PROJECT}" \
      --image "${IMAGE}" \
      --tags "${MINION_TAG}" \
      --network "${NETWORK}" \
      $2 \
      --can-ip-forward \
      --metadata-from-file "$3"; then
        if (( attempt > 5 )); then
          echo -e "${color_red}Failed to create instance template $1 ${color_norm}"
          exit 2
        fi
        echo -e "${color_yellow}Attempt $(($attempt+1)) failed to create instance template $1. Retrying.${color_norm}"
        attempt=$(($attempt+1))
    else
        break
    fi
  done
}

# Robustly try to add metadata on an instance.
#
# Arguments
#   $1: The name of the instace.
#   $2, $3, ...: The metadata key=value pair to add, where each value is a filename
function add-instance-metadata-from-file {
  local hostname=$1
  shift
  detect-project
  local attempt=0
  while true; do
    if ! gcloud compute instances add-metadata "${hostname}" \
      --project "${PROJECT}" \
      --zone "${ZONE}" \
      --metadata-from-file "$@"; then
        if (( attempt > 5 )); then
          echo -e "${color_red}Failed to add instance metadata in ${hostname} ${color_norm}"
          exit 2
        fi
        echo -e "${color_yellow}Attempt $(($attempt+1)) failed to add metadata in ${hostname}. Retrying.${color_norm}"
        attempt=$(($attempt+1))
    else
        break
    fi
  done
}

# Generates a new etcd cluster token URL.
#
# Provides:
#  KUBE_DISCOVERY_URL
function ensure-discovery-url {
  KUBE_DISCOVERY_URL=$(curl --fail -w "\n" https://discovery.etcd.io/new 2>/dev/null)
  echo "Discovery URL: ${KUBE_DISCOVERY_URL}"
}

# Initiates a local Certificate Authority (CA).
#
# Provides:
#   EASYRSA
#   EASYRSA_BIN
#   EASYRSA_EXT_DIR
#   EASYESA_SSL_CONF
function ensure-ca-certs {
  certs_dir=$1

  echo -n "Downloading easy-rsa tarball ... "
  pushd "${KUBE_TEMP}"
  curl -L -O https://storage.googleapis.com/kubernetes-release/easy-rsa/easy-rsa.tar.gz > /dev/null 2>&1
  tar xzf easy-rsa.tar.gz > /dev/null
  popd
  echo "done"
  export EASYRSA="${KUBE_TEMP}/easy-rsa-master/easyrsa3"
  export EASYRSA_EXT_DIR="${KUBE_TEMP}/easy-rsa-master/easyrsa3/x509-types"
  export EASYRSA_SSL_CONF="${KUBE_TEMP}/easy-rsa-master/easyrsa3/openssl-1.0.cnf"
  EASYRSA_BIN="${KUBE_TEMP}/easy-rsa-master/easyrsa3/easyrsa"

  echo -n "Generating CA certs ... "
  "${EASYRSA_BIN}" --pki-dir="${certs_dir}" init-pki > /dev/null 2>&1
  "${EASYRSA_BIN}" --pki-dir="${certs_dir}" --batch "--req-cn=${INSTANCE_PREFIX}" build-ca nopass > /dev/null 2>&1
  echo "done"
  # The following line is probably very hacky, but easyrsa can generate
  # server-only and client-only certs.  For secure etcd peering, we need a set
  # of certs that can be used by a server as well as a client.  We achieve this
  # by appending to "extendedKeyUsage" in the server template of easyrsa.
  sed -i -e 's/serverAuth/serverAuth,clientAuth/g' "${EASYRSA_EXT_DIR}/server"
}

# Generate TLS certs for the given hostname using the specified IP address.
#
# Assumes:
#   EASYRSA_BIN
function ensure-server-certs {
  ipaddr=${1}
  hostname=${2}
  certs_dir="$3"
  echo -n "Generating certs for host: ${hostname}, IP: ${ipaddr} ... "
  "${EASYRSA_BIN}" --pki-dir="${certs_dir}" --subject-alt-name=${ipaddr} build-server-full ${hostname} nopass > /dev/null 2>&1
  tar -czf ${local_certs_dir}/${hostname}.tar.gz \
      -C ${local_certs_dir} ca.crt issued/${hostname}.crt private/${hostname}.key
  echo "done"
}

# Generate client TLS certs.
#
# Assumes:
#   EASYRSA_BIN
function ensure-client-certs {
  certs_dir="$1"
  echo -n "Generating client certs ... "
  "${EASYRSA_BIN}" --pki-dir="${certs_dir}" build-client-full kubecfg nopass >/dev/null 2>&1
  echo "done"
}

# scp the tarball containing server keys, server certs and CA certs to the server.
#
# Assumes:
#   PROJECT
#   ZONE
function copy-server-certs {
  hostname=${1}
  remote_certs_dir=${2}
  local_certs_dir=${3}
  echo -n "Copying certs over to ${hostname} ... "
  local attempt=0
  while true; do
    sleep 10
    if ! gcloud compute copy-files --project="${PROJECT}" --zone "${ZONE}" \
      "${local_certs_dir}/${hostname}.tar.gz" root@${hostname}:/etc; then
      if (( attempt > 5 )); then
        echo
        echo -r "${color_red}Failed to copy certs over to ${hostname} ${color_norm}"
        exit 2
      fi
      echo -n -e "${color_yellow} [attempt $(($attempt+1)) failed, retrying] ${color_norm}"
      attempt=$(($attempt+1))
    else
      echo "done"
      break
    fi
  done
}

# Arguments:
#   i (Specifying the i'th minion)
#   hostname of the i'th minion
#
# Assumes:
#   KUBE_TEMP
#   MASTER_NAME
#   MINION_IP_RANGES
#   SERVER_BINARY_TAR_URL
#   KUBE_MASTER_INTERNAL_IP
#
# Provides:
#   KUBERNETES_MINION_PARAMS_TMP[$i] and related file.
function ensure-minion-i-metadata {
  ensure-temp-dir
  i="$1"
  local minion_name="$2"
  KUBERNETES_MINION_PARAMS_TMP[$i]="${KUBE_TEMP}/kubernetes-minion-params-${i}"
  (
    echo "#! /bin/bash"
    echo "MASTER_NAME='${MASTER_NAME}'"
    echo "KUBE_MINION_HOSTNAME=${minion_name}"
    echo "MINION_IP_RANGE='${MINION_IP_RANGES[$i]}'"
    echo "EXTRA_DOCKER_OPTS='${EXTRA_DOCKER_OPTS}'"
    echo "ENABLE_DOCKER_REGISTRY_CACHE='${ENABLE_DOCKER_REGISTRY_CACHE:-false}'"
    echo "SERVER_BINARY_TAR_URL='${SERVER_BINARY_TAR_URL}'"
    echo "KUBE_MASTER_INTERNAL_IP=${KUBE_MASTER_INTERNAL_IP}"
    echo "FLEET_METADATA='role=node'"
    echo "FLEET_ETCD_SERVERS=http://127.0.0.1:4001"
    echo "KUBELET_ETCD_SERVERS=http://127.0.0.1:4001"
    echo "ETCD_DISCOVERY=${KUBE_DISCOVERY_URL}"
  ) > "${KUBERNETES_MINION_PARAMS_TMP[$i]}"
}

# Prereqs:
#   upload-server-tars
#   ensure-temp-dir
# Provides:
#   KUBERNETES_MASTER_PARAMS_TMP and related file.
function ensure-master-metadata {
  export KUBERNETES_MASTER_PARAMS_TMP="${KUBE_TEMP}/kubernetes-master-params"
  (
    echo "MASTER_NAME='${MASTER_NAME}'"
    echo "NODE_INSTANCE_PREFIX='${INSTANCE_PREFIX}-minion'"
    echo "SERVER_BINARY_TAR_URL='${SERVER_BINARY_TAR_URL}'"
    echo "PORTAL_NET='${PORTAL_NET}'"
    echo "KUBE_APISERVER_SECURE_PORT='${KUBE_APISERVER_SECURE_PORT}'"
    echo "FLEET_METADATA='role=master'"
    echo "FLEET_ETCD_SERVERS=http://127.0.0.1:4001"
    echo "ETCD_DISCOVERY=${KUBE_DISCOVERY_URL}"
  ) > "$KUBERNETES_MASTER_PARAMS_TMP"
}

function restart-node () {
  local hostname=$1
  echo -n "Restarting ${hostname} ... "
  # XXX Using "gcompute instances stop/start" works for the master, but it wipes
  # the metadata for minions.  Why?
  gcloud compute ssh --project="${PROJECT}" --zone "${ZONE}" "${hostname}" \
      --command "sudo reboot" 2> /dev/null || :
  echo "done"
}

# Instantiate a kubernetes cluster
#
# Assumed vars
#   KUBE_ROOT
#   <Various vars set in config file>
function kube-up {
  detect-project

  # Make sure we have the tar files staged on Google Storage
  find-release-tars
  upload-server-tars

  ensure-temp-dir

  get-password
  python "${KUBE_ROOT}/third_party/htpasswd/htpasswd.py" \
    -b -c "${KUBE_TEMP}/htpasswd" "$KUBE_USER" "$KUBE_PASSWORD"
  local htpasswd
  htpasswd=$(cat "${KUBE_TEMP}/htpasswd")

  if ! gcloud compute networks --project "${PROJECT}" describe "${NETWORK}" &>/dev/null; then
    echo "Creating new network: ${NETWORK}"
    # The network needs to be created synchronously or we have a race. The
    # firewalls can be added concurrent with instance creation.
    gcloud compute networks create --project "${PROJECT}" "${NETWORK}" --range "10.240.0.0/16"
  fi

  if ! gcloud compute firewall-rules --project "${PROJECT}" describe "${NETWORK}-default-internal" &>/dev/null; then
    gcloud compute firewall-rules create "${NETWORK}-default-internal" \
      --project "${PROJECT}" \
      --network "${NETWORK}" \
      --source-ranges "10.0.0.0/8" \
      --allow "tcp:1-65535" "udp:1-65535" "icmp" &
  fi

  if ! gcloud compute firewall-rules describe --project "${PROJECT}" "${NETWORK}-default-ssh" &>/dev/null; then
    gcloud compute firewall-rules create "${NETWORK}-default-ssh" \
      --project "${PROJECT}" \
      --network "${NETWORK}" \
      --source-ranges "0.0.0.0/0" \
      --allow "tcp:22" &
  fi

  echo "Starting VMs and configuring firewalls"
  gcloud compute firewall-rules create "${MASTER_NAME}-https" \
    --project "${PROJECT}" \
    --network "${NETWORK}" \
    --target-tags "${MASTER_TAG}" \
    --allow tcp:443 &

  # We have to make sure the disk is created before creating the master VM, so
  # run this in the foreground.
  gcloud compute disks create "${MASTER_NAME}-pd" \
    --project "${PROJECT}" \
    --zone "${ZONE}" \
    --size "10GB" || :

  # TODO The "master-pd" disk is attached to the VM, but it is not actually
  # used (i.e., no format and mount script) yet. Add it.
  gcloud compute instances create "${MASTER_NAME}" \
    --project "${PROJECT}" \
    --zone "${ZONE}" \
    --machine-type "${MASTER_SIZE}" \
    --image-project="${IMAGE_PROJECT}" \
    --image "${IMAGE}" \
    --tags "${MASTER_TAG}" \
    --network "${NETWORK}" \
    --scopes "storage-ro" "compute-rw" \
    --disk name="${MASTER_NAME}-pd" device-name=master-pd mode=rw boot=no auto-delete=no &

  # Create a single firewall rule for all minions.
  create-firewall-rule "${MINION_TAG}-all" "${CLUSTER_IP_RANGE}" "${MINION_TAG}" &

  # Report logging choice (if any).
  if [[ "${ENABLE_NODE_LOGGING-}" == "true" ]]; then
    echo "+++ Logging using Fluentd to ${LOGGING_DESTINATION:-unknown}"
    # For logging to GCP we need to enable some minion scopes.
    if [[ "${LOGGING_DESTINATION-}" == "gcp" ]]; then
      MINION_SCOPES+=('https://www.googleapis.com/auth/logging.write')
    fi
  fi

  # Wait for last batch of jobs.
  wait-for-jobs
  detect-master

  # Set up TLS certs for the master.
  local context="${PROJECT}_${INSTANCE_PREFIX}"
  local config_dir="${HOME}/.kube/${context}"
  local local_certs_dir="${config_dir}/pki"
  local remote_certs_dir="/opt/kubernetes/.kube"
  ensure-ca-certs ${local_certs_dir}
  ensure-server-certs "IP:${KUBE_MASTER_INTERNAL_IP},IP:${KUBE_MASTER_IP}" ${MASTER_NAME} "${local_certs_dir}"
  copy-server-certs ${MASTER_NAME} ${remote_certs_dir} ${local_certs_dir}

  # Generate metadata for the master, upload and reboot.
  ensure-discovery-url
  ensure-master-metadata
  add-instance-metadata-from-file "${MASTER_NAME}" \
      "kubernetes-node-params=${KUBERNETES_MASTER_PARAMS_TMP}" \
      "user-data=${KUBE_ROOT}/cluster/gce/coreos/master.yaml"
  restart-node "${MASTER_NAME}"

  local -a scope_flags=()
  if (( "${#MINION_SCOPES[@]}" > 0 )); then
    scope_flags=("--scopes" "${MINION_SCOPES[@]}")
  else
    scope_flags=("--no-scopes")
  fi

  create-node-template "${NODE_INSTANCE_PREFIX}-template" "${scope_flags[*]}" \
      "container-bridge-sh=${KUBE_ROOT}/cluster/gce/coreos/container_bridge.sh"

  gcloud preview managed-instance-groups --zone "${ZONE}" \
      create "${NODE_INSTANCE_PREFIX}-group" \
      --project "${PROJECT}" \
      --base-instance-name "${NODE_INSTANCE_PREFIX}" \
      --size "${NUM_MINIONS}" \
      --template "${NODE_INSTANCE_PREFIX}-template" || true;
  # TODO: this should be true when the above create managed-instance-group
  # command returns, but currently it returns before the instances come up due
  # to gcloud's deficiency.
  wait-for-minions-to-run

  detect-minion-names

  # Create the routes and set IP ranges to instance metadata, 5 instances at a time.
  for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
    local minion_name="${MINION_NAMES[$i]}"
    create-route "${minion_name}" "${MINION_IP_RANGES[$i]}" &

    local minion_internal_ip=$(gcloud compute instances describe \
      --project "${PROJECT}" --zone "${ZONE}" "${minion_name}" \
      --fields networkInterfaces[0].networkIP --format=text \
      | awk '{ print $2 }')
    ensure-server-certs "IP:${minion_internal_ip}" ${minion_name} ${local_certs_dir}
    copy-server-certs ${minion_name} ${remote_certs_dir} ${local_certs_dir}
    ensure-minion-i-metadata $i ${minion_name}
    add-instance-metadata-from-file "${minion_name}" \
        "kubernetes-node-params=${KUBERNETES_MINION_PARAMS_TMP[$i]}" \
        "user-data=${KUBE_ROOT}/cluster/gce/coreos/minion.yaml"
    restart-node ${minion_name}

    if [ $i -ne 0 ] && [ $((i%5)) -eq 0 ]; then
      echo Waiting for a batch of routes at $i...
      wait-for-jobs
    fi

  done
  # Wait for last batch of jobs.
  wait-for-jobs

  # Reserve the master's IP so that it can later be transferred to another VM
  # without disrupting the kubelets. IPs are associated with regions, not zones,
  # so extract the region name, which is the same as the zone but with the final
  # dash and characters trailing the dash removed.
  local REGION=${ZONE%-*}
  gcloud compute addresses create "${MASTER_NAME}-ip" \
    --project "${PROJECT}" \
    --addresses "${KUBE_MASTER_IP}" \
    --region "${REGION}"

  echo "Waiting for cluster initialization."
  echo
  echo "  This will continually check to see if the API for kubernetes is reachable."
  echo "  This might loop forever if there was some uncaught error during start"
  echo "  up."
  echo

  until curl --insecure --user "${KUBE_USER}:${KUBE_PASSWORD}" --max-time 5 \
          --fail --output /dev/null --silent "https://${KUBE_MASTER_IP}/api/v1beta1/pods"; do
      printf "."
      sleep 2
  done

  echo "Kubernetes cluster created."

  ensure-client-certs "${local_certs_dir}"
  local kube_cert="${local_certs_dir}/issued/kubecfg.crt"
  local kube_key="${local_certs_dir}/private/kubecfg.key"
  local ca_cert="${local_certs_dir}/ca.crt"
  # TODO use token instead of kube_auth
  local kube_auth="kubernetes_auth"

  local kubectl="${KUBE_ROOT}/cluster/kubectl.sh"
  local user="${context}-admin"

  # TODO: generate ADMIN (and KUBELET) tokens and put those in the master's
  # config file.  Distribute the same way the htpasswd is done.
  (
   mkdir -p "${config_dir}"
   "${kubectl}" config set-cluster "${context}" --server="https://${KUBE_MASTER_IP}" \
       --certificate-authority="${ca_cert}" --global
   "${kubectl}" config set-credentials "${user}" --auth-path="${config_dir}/${kube_auth}" --global
   "${kubectl}" config set-context "${context}" --cluster="${context}" --user="${user}" --global
   "${kubectl}" config use-context "${context}" --global

   cat << EOF > "${config_dir}/${kube_auth}"
{
  "User": "$KUBE_USER",
  "Password": "$KUBE_PASSWORD",
  "CAFile": "${ca_cert}",
  "CertFile": "${kube_cert}",
  "KeyFile": "${kube_key}"
}
EOF

   chmod 0600 "${config_dir}/${kube_auth}" "${kube_cert}" "${kube_key}" "${ca_cert}"
   echo "Wrote ${config_dir}/${kube_auth}"
  )

  echo "Sanity checking cluster..."

  # Basic sanity checking
  # TODO Improve cluster sanity checking.
  local i
  for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
    # Make sure docker is installed and working.
    local attempt=0
    while true; do
      echo -n Attempt "$(($attempt+1))" to check Docker on node "${MINION_NAMES[$i]}" ...
      local output=$(gcloud compute --project "${PROJECT}" ssh --zone "$ZONE" "${MINION_NAMES[$i]}" --command "which docker" 2>/dev/null)
      if [[ -z "${output}" ]]; then
        if (( "$attempt" > "9" )); then
          echo
          echo -e "${color_red}Docker failed to install on node ${MINION_NAMES[$i]}. Your cluster is unlikely" >&2
          echo "to work correctly. Please run ./cluster/kube-down.sh and re-create the" >&2
          echo -e "cluster. (sorry!)${color_norm}" >&2
          exit 1
        fi
        echo -e " ${color_yellow}[not working yet]${color_norm}"
        attempt=$(($attempt+1))
        sleep 10
      else
        echo -e "${color_yellow}[${output}]${color_norm}"
        break
      fi
    done
  done
  echo
  echo -e "${color_green}Kubernetes cluster is running.  The master is running at:"
  echo
  echo -e "${color_yellow}  https://${KUBE_MASTER_IP}"
  echo
  echo -e "${color_green}The user name and password to use is located in ${config_dir}/${kube_auth}.${color_norm}"
  echo

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
function kube-down {
  detect-project

  echo "Bringing down cluster"

  gcloud preview managed-instance-groups --zone "${ZONE}" delete \
    --project "${PROJECT}" \
    --quiet \
    "${NODE_INSTANCE_PREFIX}-group" || true

  gcloud compute instance-templates delete \
    --project "${PROJECT}" \
    --quiet \
    "${NODE_INSTANCE_PREFIX}-template" || true

  # First delete the master (if it exists).
  gcloud compute instances delete \
    --project "${PROJECT}" \
    --quiet \
    --delete-disks all \
    --zone "${ZONE}" \
    "${MASTER_NAME}" || true
  # Find out what minions are running.
  local -a minions
  minions=( $(gcloud compute instances list \
                --project "${PROJECT}" --zone "${ZONE}" \
                --regexp "${NODE_INSTANCE_PREFIX}-.+" \
                | awk 'NR >= 2 { print $1 }') )
  # If any minions are running, delete them in batches.
  while (( "${#minions[@]}" > 0 )); do
    echo Deleting nodes "${minions[*]::10}"
    gcloud compute instances delete \
      --project "${PROJECT}" \
      --quiet \
      --delete-disks boot \
      --zone "${ZONE}" \
      "${minions[@]::10}" || true
    minions=( "${minions[@]:10}" )
  done

  # Delete firewall rule for the master.
  gcloud compute firewall-rules delete  \
    --project "${PROJECT}" \
    --quiet \
    "${MASTER_NAME}-https" || true

  # Delete firewall rule for minions.
  gcloud compute firewall-rules delete  \
    --project "${PROJECT}" \
    --quiet \
    "${MINION_TAG}-all" || true

  # Delete routes.
  local -a routes
  routes=( $(gcloud compute routes list --project "${PROJECT}" \
              --regexp "${NODE_INSTANCE_PREFIX}-.+" | awk 'NR >= 2 { print $1 }') )
  while (( "${#routes[@]}" > 0 )); do
    echo Deleting routes "${routes[*]::10}"
    gcloud compute routes delete \
      --project "${PROJECT}" \
      --quiet \
      "${routes[@]::10}" || true
    routes=( "${routes[@]:10}" )
  done

  # Delete the master's reserved IP
  local REGION=${ZONE%-*}
  gcloud compute addresses delete \
    --project "${PROJECT}" \
    --region "${REGION}" \
    --quiet \
    "${MASTER_NAME}-ip" || true

  # XXX OK to delete TLS certs that were created during kube-up?
  local context="${PROJECT}_${INSTANCE_PREFIX}"
  local config_dir="${HOME}/.kube/${context}"
  local local_certs_dir="${config_dir}/pki"
  rm -rf "${local_certs_dir}"
}

# Update a kubernetes cluster with latest source
function kube-push {
  echo "TODO: Support kube-push for k8s clusters based on CoreOS."
  detect-project
  detect-master

  echo
  echo "Kubernetes cluster is running.  The master is running at:"
  echo
  echo "  https://${KUBE_APISERVER_IP}"
  echo
  echo "The user name and password to use is located in ~/.kubernetes_auth."
  echo

}

# -----------------------------------------------------------------------------
# Cluster specific test helpers used from hack/e2e-test.sh

# Execute prior to running tests to build a release if required for env.
#
# Assumed Vars:
#   KUBE_ROOT
function test-build-release {
  # Make a release
  "${KUBE_ROOT}/build/release.sh"
}

# Execute prior to running tests to initialize required structure. This is
# called from hack/e2e.go only when running -up (it is run after kube-up).
#
# Assumed vars:
#   Variables from config.sh
function test-setup {
  # Detect the project into $PROJECT if it isn't set
  detect-project

  # Open up port 80 & 8080 so common containers on minions can be reached
  # TODO(roberthbailey): Remove this once we are no longer relying on hostPorts.
  gcloud compute firewall-rules create \
    --project "${PROJECT}" \
    --target-tags "${MINION_TAG}" \
    --allow tcp:80 tcp:8080 \
    --network "${NETWORK}" \
    "${MINION_TAG}-${INSTANCE_PREFIX}-http-alt"
}

# Execute after running tests to perform any required clean-up. This is called
# from hack/e2e.go
function test-teardown {
  detect-project
  echo "Shutting down test cluster in background."
  gcloud compute firewall-rules delete  \
    --project "${PROJECT}" \
    --quiet \
    "${MINION_TAG}-${INSTANCE_PREFIX}-http-alt" || true
  "${KUBE_ROOT}/cluster/kube-down.sh"
}

# SSH to a node by name ($1) and run a command ($2).
function ssh-to-node {
  local node="$1"
  local cmd="$2"
  for try in $(seq 1 5); do
    if gcloud compute ssh --ssh-flag="-o LogLevel=quiet" --project "${PROJECT}" --zone="${ZONE}" "${node}" --command "${cmd}"; then
      break
    fi
  done
}

# Restart the kube-proxy on a node ($1)
function restart-kube-proxy {
  ssh-to-node "$1" "sudo systemctl restart kube-proxy.service"
}

# Restart the kube-apiserver on a node ($1)
function restart-apiserver {
  ssh-to-node "$1" "sudo systemctl restart kube-apiserver.service"
}

# Setup monitoring firewalls using heapster and InfluxDB
function setup-monitoring-firewall {
  if [[ "${ENABLE_CLUSTER_MONITORING}" != "true" ]]; then
    return
  fi

  echo "Setting up firewalls to Heapster based cluster monitoring."

  detect-project
  gcloud compute firewall-rules create "${INSTANCE_PREFIX}-monitoring-heapster" --project "${PROJECT}" \
    --allow tcp:80 tcp:8083 tcp:8086 --target-tags="${MINION_TAG}" --network="${NETWORK}"

  echo
  echo -e "${color_green}Grafana dashboard will be available at ${color_yellow}https://${KUBE_MASTER_IP}/api/v1beta1/proxy/services/monitoring-grafana/${color_green}. Wait for the monitoring dashboard to be online.${color_norm}"
  echo
}

function teardown-monitoring-firewall {
  if [[ "${ENABLE_CLUSTER_MONITORING}" != "true" ]]; then
    return
  fi

  detect-project
  gcloud compute firewall-rules delete -q "${INSTANCE_PREFIX}-monitoring-heapster" --project "${PROJECT}" || true
}

function setup-logging-firewall {
  # If logging with Fluentd to Elasticsearch is enabled then create pods
  # and services for Elasticsearch (for ingesting logs) and Kibana (for
  # viewing logs).
  if [[ "${ENABLE_NODE_LOGGING-}" != "true" ]] || \
     [[ "${LOGGING_DESTINATION-}" != "elasticsearch" ]] || \
     [[ "${ENABLE_CLUSTER_LOGGING-}" != "true" ]]; then
    return
  fi

  detect-project
  gcloud compute firewall-rules create "${INSTANCE_PREFIX}-fluentd-elasticsearch-logging" --project "${PROJECT}" \
    --allow tcp:5601 tcp:9200 tcp:9300 --target-tags "${MINION_TAG}" --network="${NETWORK}"

  # This should be nearly instant once kube-addons gets a chance to
  # run, and we already know we can hit the apiserver, but it's still
  # worth checking.
  echo "waiting for logging services to be created by the master."
  local kubectl="${KUBE_ROOT}/cluster/kubectl.sh"
  for i in `seq 1 10`; do
    if "${kubectl}" get services -l name=kibana-logging -o template -t {{range.items}}{{.id}}{{end}} | grep -q kibana-logging &&
      "${kubectl}" get services -l name=elasticsearch-logging -o template -t {{range.items}}{{.id}}{{end}} | grep -q elasticsearch-logging; then
      break
    fi
    sleep 10
  done

  echo
  echo -e "${color_green}Cluster logs are ingested into Elasticsearch running at ${color_yellow}https://${KUBE_MASTER_IP}/api/v1beta1/proxy/services/elasticsearch-logging/"
  echo -e "${color_green}Kibana logging dashboard will be available at ${color_yellow}https://${KUBE_MASTER_IP}/api/v1beta1/proxy/services/kibana-logging/${color_norm} (note the trailing slash)"
  echo
}

function teardown-logging-firewall {
  if [[ "${ENABLE_NODE_LOGGING-}" != "true" ]] || \
     [[ "${LOGGING_DESTINATION-}" != "elasticsearch" ]] || \
     [[ "${ENABLE_CLUSTER_LOGGING-}" != "true" ]]; then
    return
  fi

  detect-project
  gcloud compute firewall-rules delete -q "${INSTANCE_PREFIX}-fluentd-elasticsearch-logging" --project "${PROJECT}" || true
  # Also delete the logging services which will remove the associated forwarding rules (TCP load balancers).
  local kubectl="${KUBE_ROOT}/cluster/kubectl.sh"
  "${kubectl}" delete services elasticsearch-logging || true
  "${kubectl}" delete services kibana-logging || true
}

# Perform preparations required to run e2e tests
function prepare-e2e() {
  detect-project
}
