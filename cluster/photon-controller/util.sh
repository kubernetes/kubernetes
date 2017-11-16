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

set -o errexit
set -o nounset
set -o pipefail

# A library of helper functions that each provider hosting Kubernetes must implement to use cluster/kube-*.sh scripts.

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/../..
# shellcheck source=./config-common.sh
source "${KUBE_ROOT}/cluster/photon-controller/config-common.sh"
# shellcheck source=./config-default.sh
source "${KUBE_ROOT}/cluster/photon-controller/${KUBE_CONFIG_FILE-"config-default.sh"}"
# shellcheck source=../common.sh
source "${KUBE_ROOT}/cluster/common.sh"

readonly PHOTON="photon -n"

# Naming scheme for VMs (masters & nodes)
readonly MASTER_NAME="${INSTANCE_PREFIX}-master"

# shell check claims this doesn't work because you can't use a variable in a brace
# range. It does work because we're calling eval.
# shellcheck disable=SC2051
readonly NODE_NAMES=($(eval echo "${INSTANCE_PREFIX}"-node-{1.."${NUM_NODES}"}))

#####################################################################
#
# Public API
#
#####################################################################

#
# detect-master will query Photon Controller for the Kubernetes master.
# It assumes that the VM name for the master is unique.
# It will set KUBE_MASTER_ID to be the VM ID of the master
# It will set KUBE_MASTER_IP to be the IP address of the master
# If the silent parameter is passed, it will not print when the master
# is found: this is used internally just to find the MASTER
#
function detect-master {
  local silent=${1:-""}
  local tenant_args="--tenant ${PHOTON_TENANT} --project ${PHOTON_PROJECT}"

  KUBE_MASTER=${MASTER_NAME}
  KUBE_MASTER_ID=${KUBE_MASTER_ID:-""}
  KUBE_MASTER_IP=${KUBE_MASTER_IP:-""}

  # We don't want silent failure: we check for failure
  set +o pipefail
  if [[ -z ${KUBE_MASTER_ID} ]]; then
    KUBE_MASTER_ID=$(${PHOTON} vm list ${tenant_args} | grep $'\t'"kubernetes-master"$'\t' | awk '{print $1}')
  fi
  if [[ -z ${KUBE_MASTER_ID} ]]; then
    kube::log::error "Could not find Kubernetes master node ID. Make sure you've launched a cluster with kube-up.sh"
    exit 1
  fi

  if [[ -z "${KUBE_MASTER_IP-}" ]]; then
      # Pick out the NICs that have a MAC address owned VMware (with OUI 00:0C:29)
      # Make sure to ignore lines that have a network interface but no address
    KUBE_MASTER_IP=$(${PHOTON} vm networks "${KUBE_MASTER_ID}" | grep -i $'\t'"00:0C:29" | grep -E '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | head -1 | awk -F'\t' '{print $3}')
  fi
  if [[ -z "${KUBE_MASTER_IP-}" ]]; then
    kube::log::error "Could not find Kubernetes master node IP. Make sure you've launched a cluster with 'kube-up.sh'" >&2
    exit 1
  fi
  if [[ -z ${silent} ]]; then
    kube::log::status "Master: $KUBE_MASTER ($KUBE_MASTER_IP)"
  fi
  # Reset default set in common.sh
  set -o pipefail
}

#
# detect-nodes will query Photon Controller for the Kubernetes nodes
# It assumes that the VM name for the nodes are unique.
# It assumes that NODE_NAMES has been set
# It will set KUBE_NODE_IP_ADDRESSES to be the VM IPs of the nodes
# It will set the KUBE_NODE_IDS to be the VM IDs of the nodes
# If the silent parameter is passed, it will not print when the nodes
# are found: this is used internally just to find the MASTER
#
function detect-nodes {
  local silent=${1:-""}
  local failure=0
  local tenant_args="--tenant ${PHOTON_TENANT} --project ${PHOTON_PROJECT}"

  KUBE_NODE_IP_ADDRESSES=()
  KUBE_NODE_IDS=()
  # We don't want silent failure: we check for failure
  set +o pipefail
  for (( i=0; i<${#NODE_NAMES[@]}; i++)); do

    local node_id
    node_id=$(${PHOTON} vm list ${tenant_args} | grep $'\t'"${NODE_NAMES[${i}]}"$'\t' | awk '{print $1}')
    if [[ -z ${node_id} ]]; then
      kube::log::error "Could not find ${NODE_NAMES[${i}]}"
      failure=1
    fi
    KUBE_NODE_IDS+=("${node_id}")

    # Pick out the NICs that have a MAC address owned VMware (with OUI 00:0C:29)
    # Make sure to ignore lines that have a network interface but no address
    node_ip=$(${PHOTON} vm networks "${node_id}" | grep -i $'\t'"00:0C:29" | grep -E '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | head -1 | awk -F'\t' '{print $3}')
    KUBE_NODE_IP_ADDRESSES+=("${node_ip}")

    if [[ -z ${silent} ]]; then
      kube::log::status "Node: ${NODE_NAMES[${i}]} (${KUBE_NODE_IP_ADDRESSES[${i}]})"
    fi
  done

  if [[ ${failure} -ne 0 ]]; then
    exit 1
  fi
  # Reset default set in common.sh
  set -o pipefail
}

# Get node names if they are not static.
function detect-node-names {
  echo "TODO: detect-node-names" 1>&2
}

#
# Verifies that this computer has sufficient software installed
# so that it can run the rest of the script.
#
function verify-prereqs {
  verify-cmd-in-path photon
  verify-cmd-in-path ssh
  verify-cmd-in-path scp
  verify-cmd-in-path ssh-add
  verify-cmd-in-path openssl
  verify-cmd-in-path mkisofs
}

#
# The entry point for bringing up a Kubernetes cluster
#
function kube-up {
  verify-prereqs
  verify-ssh-prereqs
  verify-photon-config
  kube::util::ensure-temp-dir

  find-release-tars
  find-image-id

  load-or-gen-kube-basicauth
  gen-cloud-init-iso
  gen-master-start
  create-master-vm
  install-salt-on-master

  gen-node-start
  install-salt-on-nodes

  detect-nodes -s

  install-kubernetes-on-master
  install-kubernetes-on-nodes

  wait-master-api
  wait-node-apis

  setup-pod-routes

  copy-kube-certs
  kube::log::status "Creating kubeconfig..."
  create-kubeconfig
}

# Delete a kubernetes cluster
function kube-down {
  detect-master
  detect-nodes

  pc-delete-vm "${KUBE_MASTER}" "${KUBE_MASTER_ID}"
  for (( node=0; node<${#KUBE_NODE_IDS[@]}; node++)); do
    pc-delete-vm "${NODE_NAMES[${node}]}" "${KUBE_NODE_IDS[${node}]}"
  done
}

# Update a kubernetes cluster
function kube-push {
  echo "TODO: kube-push" 1>&2
}

# Prepare update a kubernetes component
function prepare-push {
  echo "TODO: prepare-push" 1>&2
}

# Update a kubernetes master
function push-master {
  echo "TODO: push-master" 1>&2
}

# Update a kubernetes node
function push-node {
  echo "TODO: push-node" 1>&2
}

# Execute prior to running tests to build a release if required for env
function test-build-release {
  echo "TODO: test-build-release" 1>&2
}

# Execute prior to running tests to initialize required structure
function test-setup {
  echo "TODO: test-setup" 1>&2
}

# Execute after running tests to perform any required clean-up
function test-teardown {
  echo "TODO: test-teardown" 1>&2
}

#####################################################################
#
# Internal functions
#
#####################################################################

#
# Uses Photon Controller to make a VM
# Takes two parameters:
#   - The name of the VM (Assumed to be unique)
#   - The name of the flavor to create the VM (Assumed to be unique)
#
# It assumes that the variables in config-common.sh (PHOTON_TENANT, etc)
# are set correctly.
#
# It also assumes the cloud-init ISO has been generated
#
# When it completes, it sets two environment variables for use by the
# caller: _VM_ID (the ID of the created VM) and _VM_IP (the IP address
# of the created VM)
#
function pc-create-vm {
  local vm_name="${1}"
  local vm_flavor="${2}"
  local rc=0
  local i=0

  # Create the VM
  local tenant_args="--tenant ${PHOTON_TENANT} --project ${PHOTON_PROJECT}"
  local vm_args="--name ${vm_name} --image ${PHOTON_IMAGE_ID} --flavor ${vm_flavor}"
  local disk_args="disk-1 ${PHOTON_DISK_FLAVOR} boot=true"

  rc=0
  _VM_ID=$(${PHOTON} vm create ${tenant_args} ${vm_args} --disks "${disk_args}" 2>&1) || rc=$?
  if [[ ${rc} -ne 0 ]]; then
    kube::log::error "Failed to create VM. Error output:"
    echo "${_VM_ID}"
    exit 1
  fi
  kube::log::status "Created VM ${vm_name}: ${_VM_ID}"

  # Start the VM
  # Note that the VM has cloud-init in it, and we attach an ISO that
  # contains a user-data.txt file for cloud-init. When the VM starts,
  # cloud-init will temporarily mount the ISO and configure the VM
  # Our user-data will configure the 'kube' user and set up the ssh
  # authorized keys to allow us to ssh to the VM and do further work.
  run-cmd "${PHOTON} vm attach-iso -p ${KUBE_TEMP}/cloud-init.iso ${_VM_ID}"
  run-cmd "${PHOTON} vm start ${_VM_ID}"
  kube::log::status "Started VM ${vm_name}, waiting for network address..."

   # Wait for the VM to be started and connected to the network
  have_network=0
  for i in {1..120}; do
    # photon -n vm networks print several fields:
    # NETWORK MAC IP GATEWAY CONNECTED?
    # We wait until CONNECTED is True
    rc=0
    networks=$(${PHOTON} vm networks "${_VM_ID}") || rc=$?
    if [[ ${rc} -ne 0 ]]; then
      kube::log::error "'${PHOTON} vm networks ${_VM_ID}' failed. Error output: "
      echo "${networks}"
    fi
    networks=$(echo "${networks}" | grep True) || rc=$?
    if [[ ${rc} -eq 0 ]]; then
      have_network=1
      break;
    fi
    sleep 1
  done

  # Fail if the VM didn't come up
  if [[ ${have_network} -eq 0 ]]; then
    kube::log::error "VM ${vm_name} failed to start up: no IP was found"
    exit 1
  fi

  # Find the IP address of the VM
  _VM_IP=$(${PHOTON} vm networks "${_VM_ID}" | head -1 | awk -F'\t' '{print $3}')
  kube::log::status "VM ${vm_name} has IP: ${_VM_IP}"
}

#
# Delete one of our VMs
# If it is STARTED, it will be stopped first.
#
function pc-delete-vm {
  local vm_name="${1}"
  local vm_id="${2}"
  local rc=0

  kube::log::status "Deleting VM ${vm_name}"
  # In some cases, head exits before photon, so the pipline exits with
  # SIGPIPE. We disable the pipefile option to hide that failure.
  set +o pipefail
  ${PHOTON} vm show "${vm_id}" | head -1 | grep STARTED > /dev/null 2>&1 || rc=$?
  set +o pipefail
  if [[ ${rc} -eq 0 ]]; then
    ${PHOTON} vm stop "${vm_id}" > /dev/null 2>&1 || rc=$?
    if [[ ${rc} -ne 0 ]]; then
      kube::log::error "Error: could not stop ${vm_name} ($vm_id)"
      kube::log::error "Please investigate and stop manually"
      return
    fi
  fi

  rc=0
  ${PHOTON} vm delete "${vm_id}" > /dev/null 2>&1 || rc=$?
  if [[ ${rc} -ne 0 ]]; then
    kube::log::error "Error: could not delete ${vm_name} ($vm_id)"
    kube::log::error "Please investigate and delete manually"
  fi
}

#
# Looks for the image named PHOTON_IMAGE
# Sets PHOTON_IMAGE_ID to be the id of that image.
# We currently assume there is exactly one image with name
#
function find-image-id {
  local rc=0
  PHOTON_IMAGE_ID=$(${PHOTON} image list | grep $'\t'"${PHOTON_IMAGE}"$'\t' | head -1 | grep READY | awk -F'\t' '{print $1}')
  if [[ ${rc} -ne 0 ]]; then
    kube::log::error "Cannot find image \"${PHOTON_IMAGE}\""
    fail=1
  fi
}

#
# Generate an ISO with a single file called user-data.txt
# This ISO will be used to configure cloud-init (which is already
# on the VM). We will tell cloud-init to create the kube user/group
# and give ourselves the ability to ssh to the VM with ssh. We also
# allow people to ssh with the same password that was randomly
# generated for access to Kubernetes as a backup method.
#
# Assumes environment variables:
#   - VM_USER
#   - KUBE_PASSWORD (randomly generated password)
#
function gen-cloud-init-iso {
  local password_hash
  password_hash=$(openssl passwd -1 "${KUBE_PASSWORD}")

  local ssh_key
  ssh_key=$(ssh-add -L | head -1)

  # Make the user-data file that will be used by cloud-init
  (
    echo "#cloud-config"
    echo ""
    echo "groups:"
    echo "  - ${VM_USER}"
    echo ""
    echo "users:"
    echo "  - name: ${VM_USER}"
    echo "    gecos: Kubernetes"
    echo "    primary-group: ${VM_USER}"
    echo "    lock-passwd: false"
    echo "    passwd: ${password_hash}"
    echo "    ssh-authorized-keys: "
    echo "      - ${ssh_key}"
    echo "    sudo: ALL=(ALL) NOPASSWD:ALL"
    echo "    shell: /bin/bash"
    echo ""
    echo "hostname:"
    echo "  - hostname: kube"
  ) > "${KUBE_TEMP}/user-data.txt"

  # Make the ISO that will contain the user-data
  # The -rock option means that we'll generate real filenames (long and with case)
  run-cmd "mkisofs -rock -o ${KUBE_TEMP}/cloud-init.iso ${KUBE_TEMP}/user-data.txt"
}

#
# Generate a script used to install salt on the master
# It is placed into $KUBE_TEMP/master-start.sh
#
function gen-master-start {
  python "${KUBE_ROOT}/third_party/htpasswd/htpasswd.py" \
    -b -c "${KUBE_TEMP}/htpasswd" "${KUBE_USER}" "${KUBE_PASSWORD}"
  local htpasswd
  htpasswd=$(cat "${KUBE_TEMP}/htpasswd")

  # This calculation of the service IP should work, but if you choose an
  # alternate subnet, there's a small chance you'd need to modify the
  # service_ip, below.  We'll choose an IP like 10.244.240.1 by taking
  # the first three octets of the SERVICE_CLUSTER_IP_RANGE and tacking
  # on a .1
  local octets
  local service_ip
  octets=($(echo "${SERVICE_CLUSTER_IP_RANGE}" | sed -e 's|/.*||' -e 's/\./ /g'))
  ((octets[3]+=1))
  service_ip=$(echo "${octets[*]}" | sed 's/ /./g')
  MASTER_EXTRA_SANS="IP:${service_ip},DNS:${MASTER_NAME},${MASTER_EXTRA_SANS}"

  (
    echo "#! /bin/bash"
    echo "readonly MY_NAME=${MASTER_NAME}"
    grep -v "^#" "${KUBE_ROOT}/cluster/photon-controller/templates/hostname.sh"
    echo "cd /home/kube/cache/kubernetes-install"
    echo "readonly KUBE_MASTER_IP='{$KUBE_MASTER_IP}'"
    echo "readonly MASTER_NAME='${MASTER_NAME}'"
    echo "readonly MASTER_IP_RANGE='${MASTER_IP_RANGE}'"
    echo "readonly INSTANCE_PREFIX='${INSTANCE_PREFIX}'"
    echo "readonly NODE_INSTANCE_PREFIX='${INSTANCE_PREFIX}-node'"
    echo "readonly NODE_IP_RANGES='${NODE_IP_RANGES}'"
    echo "readonly SERVICE_CLUSTER_IP_RANGE='${SERVICE_CLUSTER_IP_RANGE}'"
    echo "readonly ENABLE_NODE_LOGGING='${ENABLE_NODE_LOGGING:-false}'"
    echo "readonly LOGGING_DESTINATION='${LOGGING_DESTINATION:-}'"
    echo "readonly ENABLE_CLUSTER_DNS='${ENABLE_CLUSTER_DNS:-false}'"
    echo "readonly ENABLE_CLUSTER_UI='${ENABLE_CLUSTER_UI:-false}'"
    echo "readonly DNS_SERVER_IP='${DNS_SERVER_IP:-}'"
    echo "readonly DNS_DOMAIN='${DNS_DOMAIN:-}'"
    echo "readonly KUBE_USER='${KUBE_USER:-}'"
    echo "readonly KUBE_PASSWORD='${KUBE_PASSWORD:-}'"
    echo "readonly SERVER_BINARY_TAR='${SERVER_BINARY_TAR##*/}'"
    echo "readonly SALT_TAR='${SALT_TAR##*/}'"
    echo "readonly MASTER_HTPASSWD='${htpasswd}'"
    echo "readonly E2E_STORAGE_TEST_ENVIRONMENT='${E2E_STORAGE_TEST_ENVIRONMENT:-}'"
    echo "readonly MASTER_EXTRA_SANS='${MASTER_EXTRA_SANS:-}'"
    grep -v "^#" "${KUBE_ROOT}/cluster/photon-controller/templates/create-dynamic-salt-files.sh"
    grep -v "^#" "${KUBE_ROOT}/cluster/photon-controller/templates/install-release.sh"
    grep -v "^#" "${KUBE_ROOT}/cluster/photon-controller/templates/salt-master.sh"
  ) > "${KUBE_TEMP}/master-start.sh"
}

#
# Generate the scripts for each node to install salt
#
function gen-node-start {
  local i
  for (( i=0; i<${#NODE_NAMES[@]}; i++)); do
    (
      echo "#! /bin/bash"
      echo "readonly MY_NAME=${NODE_NAMES[${i}]}"
      grep -v "^#" "${KUBE_ROOT}/cluster/photon-controller/templates/hostname.sh"
      echo "KUBE_MASTER=${KUBE_MASTER}"
      echo "KUBE_MASTER_IP=${KUBE_MASTER_IP}"
      echo "NODE_IP_RANGE=$NODE_IP_RANGES"
      grep -v "^#" "${KUBE_ROOT}/cluster/photon-controller/templates/salt-minion.sh"
    ) > "${KUBE_TEMP}/node-start-${i}.sh"
  done
}

#
# Create a script that will run on the Kubernetes master and will run salt
# to configure the master. We make it a script instead of just running a
# single ssh command so that we can get logging.
#
function gen-master-salt {
  gen-salt "kubernetes-master"
}

#
# Create scripts that will be run on the Kubernetes master. Each of these
# will invoke salt to configure one of the nodes
#
function gen-node-salt {
  local i
  for (( i=0; i<${#NODE_NAMES[@]}; i++)); do
    gen-salt "${NODE_NAMES[${i}]}"
  done
}

#
# Shared implementation for gen-master-salt and gen-node-salt
# Writes a script that installs Kubernetes with salt
# The core of the script is simple (run 'salt ... state.highstate')
# We also do a bit of logging so we can debug problems
#
# There is also a funky workaround for an issue with docker 1.9
# (elsewhere we peg ourselves to docker 1.9). It's fixed in 1.10,
# so we should be able to remove it in the future
# https://github.com/docker/docker/issues/18113
# The problem is that sometimes the install (with apt-get) of
# docker fails. Deleting a file and retrying fixes it.
#
# Tell shellcheck to ignore our variables within single quotes:
# We're writing a script, not executing it, so this is normal
# shellcheck disable=SC2016
function gen-salt {
  node_name=${1}
    (
      echo '#!/bin/bash'
      echo ''
      echo "node=${node_name}"
      echo 'out=/tmp/${node}-salt.out'
      echo 'log=/tmp/${node}-salt.log'
      echo ''
      echo 'echo $(date) >> $log'
      echo 'salt ${node} state.highstate -t 30 --no-color > ${out}'
      echo 'grep -E "Failed:[[:space:]]+0" ${out}'
      echo 'success=$?'
      echo 'cat ${out} >> ${log}'
      echo ''
      echo 'if [[ ${success} -ne 0 ]]; then'
      echo '  # Did we try to install docker-engine?'
      echo '  attempted=$(grep docker-engine ${out} | wc -l)'
      echo '  # Is docker-engine installed?'
      echo '  installed=$(salt --output=txt ${node} pkg.version docker-engine | wc -l)'
      echo '  if [[ ${attempted} -ne 0 && ${installed} -eq 0 ]]; then'
      echo '    echo "Unwedging docker-engine install" >> ${log}'
      echo '    salt ${node} cmd.run "rm -f /var/lib/docker/network/files/local-kv.db"'
      echo '  fi'
      echo 'fi'
      echo 'exit ${success}'
    ) > "${KUBE_TEMP}/${node_name}-salt.sh"
}

#
# Generate a script to add a route to a host (master or node)
# The script will do two things:
# 1. Add the route immediately with the route command
# 2. Persist the route by saving it in /etc/network/interfaces
# This was done with a script because it was easier to get the quoting right
# and make it clear.
#
function gen-add-route {
  route=${1}
  gateway=${2}
  (
      echo '#!/bin/bash'
      echo ''
      echo '# Immediately add route'
      echo "sudo route add -net ${route} gw ${gateway}"
      echo ''
      echo '# Persist route so it lasts over restarts'
      echo 'sed -in "s|^iface eth0.*|&\n    post-up route add -net' "${route} gw ${gateway}|"'" /etc/network/interfaces'
  ) > "${KUBE_TEMP}/add-route.sh"
}

#
# Create the Kubernetes master VM
# Sets global variables:
# - KUBE_MASTER    (Name)
# - KUBE_MASTER_ID (Photon VM ID)
# - KUBE_MASTER_IP (IP address)
#
function create-master-vm {
  kube::log::status "Starting master VM..."
  pc-create-vm "${MASTER_NAME}" "${PHOTON_MASTER_FLAVOR}"
  KUBE_MASTER=${MASTER_NAME}
  KUBE_MASTER_ID=${_VM_ID}
  KUBE_MASTER_IP=${_VM_IP}
}

#
# Install salt on the Kubernetes master
# Relies on the master-start.sh script created in gen-master-start
#
function install-salt-on-master {
  kube::log::status "Installing salt on master..."
  upload-server-tars "${MASTER_NAME}" "${KUBE_MASTER_IP}"
  run-script-remotely "${KUBE_MASTER_IP}" "${KUBE_TEMP}/master-start.sh"
}

#
# Installs salt on Kubernetes nodes in parallel
# Relies on the node-start script created in gen-node-start
#
function install-salt-on-nodes {
  kube::log::status "Creating nodes and installing salt on them..."

  # Start each of the VMs in parallel
  # In the future, we'll batch this because it doesn't scale well
  # past 10 or 20 nodes
  local node
  for (( node=0; node<${#NODE_NAMES[@]}; node++)); do
  (
    pc-create-vm "${NODE_NAMES[${node}]}" "${PHOTON_NODE_FLAVOR}"
    run-script-remotely "${_VM_IP}" "${KUBE_TEMP}/node-start-${node}.sh"
  ) &
  done

  # Wait for the node VM startups to complete
  local fail=0
  local job
  for job in $(jobs -p); do
    wait "${job}" || fail=$((fail + 1))
  done
  if (( fail != 0 )); then
    kube::log::error "Failed to start ${fail}/${NUM_NODES} nodes"
    exit 1
  fi
}

#
# Install Kubernetes on the master.
# This uses the kubernetes-master-salt.sh script created by gen-master-salt
# That script uses salt to install Kubernetes
#
function install-kubernetes-on-master {
  # Wait until salt-master is running: it may take a bit
  try-until-success-ssh "${KUBE_MASTER_IP}" \
    "Waiting for salt-master to start on ${KUBE_MASTER}" \
    "pgrep salt-master"
  gen-master-salt
  copy-file-to-vm "${_VM_IP}" "${KUBE_TEMP}/kubernetes-master-salt.sh" "/tmp/kubernetes-master-salt.sh"
  try-until-success-ssh "${KUBE_MASTER_IP}" \
    "Installing Kubernetes on ${KUBE_MASTER} via salt" \
    "sudo /bin/bash /tmp/kubernetes-master-salt.sh"
}

#
# Install Kubernetes on the nodes in parallel
# This uses the kubernetes-master-salt.sh script created by gen-node-salt
# That script uses salt to install Kubernetes
#
function install-kubernetes-on-nodes {
  gen-node-salt

  # Run in parallel to bring up the cluster faster
  # TODO: Batch this so that we run up to N in parallel, so
  # we don't overload this machine or the salt master
  local node
  for (( node=0; node<${#NODE_NAMES[@]}; node++)); do
    (
      copy-file-to-vm "${_VM_IP}" "${KUBE_TEMP}/${NODE_NAMES[${node}]}-salt.sh" "/tmp/${NODE_NAMES[${node}]}-salt.sh"
      try-until-success-ssh "${KUBE_NODE_IP_ADDRESSES[${node}]}" \
        "Waiting for salt-master to start on ${NODE_NAMES[${node}]}" \
        "pgrep salt-minion"
      try-until-success-ssh "${KUBE_MASTER_IP}" \
        "Installing Kubernetes on ${NODE_NAMES[${node}]} via salt" \
        "sudo /bin/bash /tmp/${NODE_NAMES[${node}]}-salt.sh"
    ) &
  done

  # Wait for the Kubernetes installations to complete
  local fail=0
  local job
  for job in $(jobs -p); do
    wait "${job}" || fail=$((fail + 1))
  done
  if (( fail != 0 )); then
    kube::log::error "Failed to start install Kubernetes on ${fail} out of ${NUM_NODES} nodess"
    exit 1
  fi
}

#
# Upload the Kubernetes tarballs to the master
#
function upload-server-tars {
  vm_name=${1}
  vm_ip=${2}

  run-ssh-cmd "${vm_ip}" "mkdir -p /home/kube/cache/kubernetes-install"

  local tar
  for tar in "${SERVER_BINARY_TAR}" "${SALT_TAR}"; do
    local base_tar
    base_tar=$(basename "${tar}")
    kube::log::status "Uploading ${base_tar} to ${vm_name}..."
    copy-file-to-vm "${vm_ip}" "${tar}" "/home/kube/cache/kubernetes-install/${tar##*/}"
  done
}

#
# Wait for the Kubernets healthz API to be responsive on the master
#
function wait-master-api {
  local curl_creds="--insecure --user ${KUBE_USER}:${KUBE_PASSWORD}"
  local curl_output="--fail --output /dev/null --silent"
  local curl_net="--max-time 1"

  try-until-success "Waiting for Kubernetes API on ${KUBE_MASTER}" \
    "curl ${curl_creds} ${curl_output} ${curl_net} https://${KUBE_MASTER_IP}/healthz"
}

#
# Wait for the Kubernetes healthz API to be responsive on each node
#
function wait-node-apis {
  local curl_output="--fail --output /dev/null --silent"
  local curl_net="--max-time 1"

  for (( i=0; i<${#NODE_NAMES[@]}; i++)); do
    try-until-success "Waiting for Kubernetes API on ${NODE_NAMES[${i}]}..." \
      "curl ${curl_output} ${curl_net} http://${KUBE_NODE_IP_ADDRESSES[${i}]}:10250/healthz"
  done
}

#
# Configure the nodes so the pods can communicate
# Each node will have a bridge named cbr0 for the NODE_IP_RANGES
# defined in config-default.sh. This finds the IP subnet (assigned
# by Kubernetes) to nodes and configures routes so they can communicate
#
# Also configure the master to be able to talk to the nodes. This is
# useful so that you can get to the UI from the master.
#
function setup-pod-routes {
  local node

  KUBE_NODE_BRIDGE_NETWORK=()
  for (( node=0; node<${#NODE_NAMES[@]}; node++)); do

    # This happens in two steps (wait for an address, wait for a non 172.x.x.x address)
    # because it's both simpler and more clear what's happening.
    try-until-success-ssh "${KUBE_NODE_IP_ADDRESSES[${node}]}" \
      "Waiting for cbr0 bridge on ${NODE_NAMES[${node}]} to have an address"  \
      'sudo ifconfig cbr0  | grep -oP "inet addr:\K\S+"'

    try-until-success-ssh "${KUBE_NODE_IP_ADDRESSES[${node}]}" \
      "Waiting for cbr0 bridge on ${NODE_NAMES[${node}]} to have correct address"  \
      'sudo ifconfig cbr0  | grep -oP "inet addr:\K\S+" | grep -v  "^172."'

    run-ssh-cmd "${KUBE_NODE_IP_ADDRESSES[${node}]}" 'sudo ip route show | grep -E "dev cbr0" | cut -d " " -f1'
    KUBE_NODE_BRIDGE_NETWORK+=(${_OUTPUT})
    kube::log::status "cbr0 on ${NODE_NAMES[${node}]} is ${_OUTPUT}"
  done

  local i
  local j
  for (( i=0; i<${#NODE_NAMES[@]}; i++)); do
    kube::log::status "Configuring pod routes on ${NODE_NAMES[${i}]}..."
    gen-add-route "${KUBE_NODE_BRIDGE_NETWORK[${i}]}" "${KUBE_NODE_IP_ADDRESSES[${i}]}"
    run-script-remotely "${KUBE_MASTER_IP}" "${KUBE_TEMP}/add-route.sh"

    for (( j=0; j<${#NODE_NAMES[@]}; j++)); do
      if [[ "${i}" != "${j}" ]]; then
        gen-add-route "${KUBE_NODE_BRIDGE_NETWORK[${j}]}" "${KUBE_NODE_IP_ADDRESSES[${j}]}"
        run-script-remotely "${KUBE_NODE_IP_ADDRESSES[${i}]}" "${KUBE_TEMP}/add-route.sh"
      fi
    done
  done
}

#
# Copy the certificate/key from the Kubernetes master
# These are used to create the kubeconfig file, which allows
# users to use kubectl easily
#
# We also set KUBE_CERT, KUBE_KEY, CA_CERT, and CONTEXT because they
# are needed by create-kubeconfig from common.sh to generate
# the kube config file.
#
function copy-kube-certs {
  local cert="kubecfg.crt"
  local key="kubecfg.key"
  local ca="ca.crt"
  local cert_dir="/srv/kubernetes"

  kube::log::status "Copying credentials from ${KUBE_MASTER}"

  # Set global environment variables: needed by create-kubeconfig
  # in common.sh
  export KUBE_CERT="${KUBE_TEMP}/${cert}"
  export KUBE_KEY="${KUBE_TEMP}/${key}"
  export CA_CERT="${KUBE_TEMP}/${ca}"
  export CONTEXT="photon-${INSTANCE_PREFIX}"

  run-ssh-cmd "${KUBE_MASTER_IP}" "sudo chmod 644 ${cert_dir}/${cert}"
  run-ssh-cmd "${KUBE_MASTER_IP}" "sudo chmod 644 ${cert_dir}/${key}"
  run-ssh-cmd "${KUBE_MASTER_IP}" "sudo chmod 644 ${cert_dir}/${ca}"

  copy-file-from-vm "${KUBE_MASTER_IP}" "${cert_dir}/${cert}" "${KUBE_CERT}"
  copy-file-from-vm "${KUBE_MASTER_IP}" "${cert_dir}/${key}"  "${KUBE_KEY}"
  copy-file-from-vm "${KUBE_MASTER_IP}" "${cert_dir}/${ca}"   "${CA_CERT}"

  run-ssh-cmd "${KUBE_MASTER_IP}" "sudo chmod 600 ${cert_dir}/${cert}"
  run-ssh-cmd "${KUBE_MASTER_IP}" "sudo chmod 600 ${cert_dir}/${key}"
  run-ssh-cmd "${KUBE_MASTER_IP}" "sudo chmod 600 ${cert_dir}/${ca}"
}

#
# Copies a script to a VM and runs it
# Parameters:
#   - IP of VM
#   - Path to local file
#
function run-script-remotely {
  local vm_ip=${1}
  local local_file="${2}"
  local base_file
  local remote_file

  base_file=$(basename "${local_file}")
  remote_file="/tmp/${base_file}"

  copy-file-to-vm "${vm_ip}" "${local_file}" "${remote_file}"
  run-ssh-cmd "${vm_ip}" "chmod 700 ${remote_file}"
  run-ssh-cmd "${vm_ip}" "nohup sudo ${remote_file} < /dev/null 1> ${remote_file}.out 2>&1 &"
}

#
# Runs an command on a VM using ssh
# Parameters:
#   - (optional) -i to ignore failure
#   - IP address of the VM
#   - Command to run
# Assumes environment variables:
#   - VM_USER
#   - SSH_OPTS
#
function run-ssh-cmd {
  local ignore_failure=""
  if [[ "${1}" = "-i" ]]; then
    ignore_failure="-i"
    shift
  fi

  local vm_ip=${1}
  shift
  local cmd=${1}


  run-cmd ${ignore_failure} "ssh ${SSH_OPTS} $VM_USER@${vm_ip} $1"
}

#
# Uses scp to copy file to VM
# Parameters:
#   - IP address of the VM
#   - Path to local file
#   - Path to remote file
# Assumes environment variables:
#   - VM_USER
#   - SSH_OPTS
#
function copy-file-to-vm {
  local vm_ip=${1}
  local local_file=${2}
  local remote_file=${3}

  run-cmd "scp ${SSH_OPTS} ${local_file} ${VM_USER}@${vm_ip}:${remote_file}"
}

function copy-file-from-vm {
  local vm_ip=${1}
  local remote_file=${2}
  local local_file=${3}

  run-cmd "scp ${SSH_OPTS} ${VM_USER}@${vm_ip}:${remote_file} ${local_file}"
}

#
# Run a command, print nice error output
# Used by copy-file-to-vm and run-ssh-cmd
#
function run-cmd {
  local rc=0
  local ignore_failure=""
  if [[ "${1}" = "-i" ]]; then
    ignore_failure=${1}
    shift
  fi

  local cmd=$1
  local output
  output=$(${cmd} 2>&1) || rc=$?
  if [[ ${rc} -ne 0 ]]; then
    if [[ -z "${ignore_failure}" ]]; then
      kube::log::error "Failed to run command: ${cmd} Output:"
      echo "${output}"
      exit 1
    fi
  fi
  _OUTPUT=${output}
  return ${rc}
}

#
# After the initial VM setup, we use SSH with keys to access the VMs
# This requires an SSH agent, so we verify that it's running
#
function verify-ssh-prereqs {
  kube::log::status "Validating SSH configuration..."
  local rc

  rc=0
  ssh-add -L 1> /dev/null 2> /dev/null || rc=$?
  # "Could not open a connection to your authentication agent."
  if [[ "${rc}" -eq 2 ]]; then
    # ssh agent wasn't running, so start it and ensure we stop it
    eval "$(ssh-agent)" > /dev/null
    trap-add "kill ${SSH_AGENT_PID}" EXIT
  fi

  rc=0
  ssh-add -L 1> /dev/null 2> /dev/null || rc=$?
  # "The agent has no identities."
  if [[ "${rc}" -eq 1 ]]; then
  # Try adding one of the default identities, with or without passphrase.
    ssh-add || true
  fi

  # Expect at least one identity to be available.
  if ! ssh-add -L 1> /dev/null 2> /dev/null; then
    kube::log::error "Could not find or add an SSH identity."
    kube::log::error "Please start ssh-agent, add your identity, and retry."
    exit 1
  fi
}

#
# Verify that Photon Controller has been configured in the way we expect. Specifically
# - Have the flavors been created?
# - Has the image been uploaded?
# TODO: Check the tenant and project as well.
function verify-photon-config {
  kube::log::status "Validating Photon configuration..."

  # We don't want silent failure: we check for failure
  set +o pipefail

  verify-photon-flavors
  verify-photon-image
  verify-photon-tenant

  # Reset default set in common.sh
  set -o pipefail
}

#
# Verify that the VM and disk flavors have been created
#
function verify-photon-flavors {
  local rc=0

  ${PHOTON} flavor list | awk -F'\t' '{print $2}' | grep -q "^${PHOTON_MASTER_FLAVOR}$" > /dev/null 2>&1 || rc=$?
  if [[ ${rc} -ne 0 ]]; then
    kube::log::error "ERROR: Cannot find VM flavor named ${PHOTON_MASTER_FLAVOR}"
    exit 1
  fi

  if [[ "${PHOTON_MASTER_FLAVOR}" != "${PHOTON_NODE_FLAVOR}" ]]; then
    rc=0
    ${PHOTON} flavor list | awk -F'\t' '{print $2}' | grep -q "^${PHOTON_NODE_FLAVOR}$" > /dev/null 2>&1 || rc=$?
    if [[ ${rc} -ne 0 ]]; then
      kube::log::error "ERROR: Cannot find VM flavor named ${PHOTON_NODE_FLAVOR}"
      exit 1
    fi
  fi

  ${PHOTON} flavor list | awk -F'\t' '{print $2}' | grep -q "^${PHOTON_DISK_FLAVOR}$" > /dev/null 2>&1 || rc=$?
  if [[ ${rc} -ne 0 ]]; then
    kube::log::error "ERROR: Cannot find disk flavor named ${PHOTON_DISK_FLAVOR}"
    exit 1
  fi
}

#
# Verify that we have the image we need, and it's not in error state or
# multiple copies
#
function verify-photon-image {
  local rc

  rc=0
  ${PHOTON} image list | grep -q $'\t'"${PHOTON_IMAGE}"$'\t'  > /dev/null 2>&1 || rc=$?
  if [[ ${rc} -ne 0 ]]; then
    kube::log::error "ERROR: Cannot find image \"${PHOTON_IMAGE}\""
    exit 1
  fi

  rc=0
  ${PHOTON} image list | grep $'\t'"${PHOTON_IMAGE}"$'\t' | grep ERROR > /dev/null 2>&1 || rc=$?
  if [[ ${rc} -eq 0 ]]; then
    echo "Warning: You have at least one ${PHOTON_IMAGE} image in the ERROR state. You may want to investigate."
    echo "Images in the ERROR state will be ignored."
  fi

  rc=0
  num_images=$(${PHOTON} image list | grep $'\t'"${PHOTON_IMAGE}"$'\t' | grep -c READY)
  if [[ "${num_images}" -gt 1 ]]; then
    echo "ERROR: You have more than one READY ${PHOTON_IMAGE} image. Ensure there is only one"
    exit 1
  fi
}

function verify-photon-tenant {
  local rc

  rc=0
  ${PHOTON} tenant list | grep -q $'\t'"${PHOTON_TENANT}"  > /dev/null 2>&1 || rc=$?
  if [[ ${rc} -ne 0 ]]; then
    echo "ERROR: Cannot find tenant \"${PHOTON_TENANT}\""
    exit 1
  fi

  ${PHOTON} project list --tenant "${PHOTON_TENANT}" | grep -q $'\t'"${PHOTON_PROJECT}"$'\t' > /dev/null 2>&1  || rc=$?
  if [[ ${rc} -ne 0 ]]; then
    echo "ERROR: Cannot find project \"${PHOTON_PROJECT}\""
    exit 1
  fi
}

#
# Verifies that a given command is in the PATH
#
function verify-cmd-in-path {
  cmd=${1}
  which "${cmd}" >/dev/null || {
    kube::log::error "Can't find ${cmd} in PATH, please install and retry."
    exit 1
  }
}

#
# Repeatedly try a command over ssh until it succeeds or until five minutes have passed
# The timeout isn't exact, since we assume the command runs instantaneously, and
# it doesn't.
#
function try-until-success-ssh {
  local vm_ip=${1}
  local cmd_description=${2}
  local cmd=${3}
  local timeout=600
  local sleep_time=5
  local max_attempts

  ((max_attempts=timeout/sleep_time))

  kube::log::status "${cmd_description} for up to 10 minutes..."
  local attempt=0
  while true; do
    local rc=0
    run-ssh-cmd -i "${vm_ip}" "${cmd}" || rc=1
    if [[ ${rc} != 0 ]]; then
      if (( attempt == max_attempts )); then
        kube::log::error "Failed, cannot proceed: you may need to retry to log into the VM to debug"
        exit 1
      fi
    else
      break
    fi
    attempt=$((attempt+1))
    sleep ${sleep_time}
  done
}

function try-until-success {
  local cmd_description=${1}
  local cmd=${2}
  local timeout=600
  local sleep_time=5
  local max_attempts

  ((max_attempts=timeout/sleep_time))

  kube::log::status "${cmd_description} for up to 10 minutes..."
  local attempt=0
  while true; do
    local rc=0
    run-cmd -i "${cmd}" || rc=1
    if [[ ${rc} != 0 ]]; then
      if (( attempt == max_attempts )); then
        kube::log::error "Failed, cannot proceed"
        exit 1
      fi
    else
      break
    fi
    attempt=$((attempt+1))
    sleep ${sleep_time}
  done
}

#
# Sets up a trap handler
#
function trap-add {
  local handler="${1}"
  local signal="${2-EXIT}"
  local cur

  cur="$(eval "sh -c 'echo \$3' -- $(trap -p ${signal})")"
  if [[ -n "${cur}" ]]; then
    handler="${cur}; ${handler}"
  fi

  # We want ${handler} to expand now, so tell shellcheck
  # shellcheck disable=SC2064
  trap "${handler}" ${signal}
}
