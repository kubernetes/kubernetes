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

# A library of helper functions that each provider hosting Kubernetes must implement to use cluster/kube-*.sh scripts.

# exit on any error
set -e

SSH_OPTS="-oStrictHostKeyChecking=no -oUserKnownHostsFile=/dev/null -oLogLevel=ERROR -C"

# Use the config file specified in $KUBE_CONFIG_FILE, or default to
# config-default.sh.
KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
readonly ROOT=$(dirname "${BASH_SOURCE}")
source "${ROOT}/${KUBE_CONFIG_FILE:-"config-default.sh"}"
source "$KUBE_ROOT/cluster/common.sh"


KUBECTL_PATH=${KUBE_ROOT}/cluster/centos/binaries/kubectl

# Directory to be used for master and node provisioning.
KUBE_TEMP="~/kube_temp"


# Get master IP addresses and store in KUBE_MASTER_IP_ADDRESSES[]
# Must ensure that the following ENV vars are set:
#   MASTERS
function detect-masters() {
  KUBE_MASTER_IP_ADDRESSES=()
  for master in ${MASTERS}; do
    KUBE_MASTER_IP_ADDRESSES+=("${master#*@}")
  done
  echo "KUBE_MASTERS: ${MASTERS}" 1>&2
  echo "KUBE_MASTER_IP_ADDRESSES: [${KUBE_MASTER_IP_ADDRESSES[*]}]" 1>&2
}

# Get node IP addresses and store in KUBE_NODE_IP_ADDRESSES[]
function detect-nodes() {
  KUBE_NODE_IP_ADDRESSES=()
  for node in ${NODES}; do
    KUBE_NODE_IP_ADDRESSES+=("${node#*@}")
  done
  echo "KUBE_NODE_IP_ADDRESSES: [${KUBE_NODE_IP_ADDRESSES[*]}]" 1>&2
}

# Verify prereqs on host machine
function verify-prereqs() {
  local rc
  rc=0
  ssh-add -L 1> /dev/null 2> /dev/null || rc="$?"
  # "Could not open a connection to your authentication agent."
  if [[ "${rc}" -eq 2 ]]; then
    eval "$(ssh-agent)" > /dev/null
    trap-add "kill ${SSH_AGENT_PID}" EXIT
  fi
  rc=0
  ssh-add -L 1> /dev/null 2> /dev/null || rc="$?"
  # "The agent has no identities."
  if [[ "${rc}" -eq 1 ]]; then
    # Try adding one of the default identities, with or without passphrase.
    ssh-add || true
  fi
  rc=0
  # Expect at least one identity to be available.
  if ! ssh-add -L 1> /dev/null 2> /dev/null; then
    echo "Could not find or add an SSH identity."
    echo "Please start ssh-agent, add your identity, and retry."
    exit 1
  fi
}

# Install handler for signal trap
function trap-add {
  local handler="$1"
  local signal="${2-EXIT}"
  local cur

  cur="$(eval "sh -c 'echo \$3' -- $(trap -p ${signal})")"
  if [[ -n "${cur}" ]]; then
    handler="${cur}; ${handler}"
  fi

  trap "${handler}" ${signal}
}

# Validate a kubernetes cluster
function validate-cluster() {
  # by default call the generic validate-cluster.sh script, customizable by
  # any cluster provider if this does not fit.
  set +e
  "${KUBE_ROOT}/cluster/validate-cluster.sh"
  if [[ "$?" -ne "0" ]]; then
    for master in ${MASTERS}; do
      troubleshoot-master ${master}
    done
    for node in ${NODES}; do
      troubleshoot-node ${node}
    done
    exit 1
  fi
  set -e
}

# Instantiate a kubernetes cluster
function kube-up() {
  make-ca-cert

  local num_infra=0
  for master in ${MASTERS}; do
    provision-master "${master}" "infra${num_infra}"
    let ++num_infra
  done

  for master in ${MASTERS}; do
    post-provision-master "${master}"
  done

  for node in ${NODES}; do
    provision-node "${node}"
  done

  detect-masters

  # set CONTEXT and KUBE_SERVER values for create-kubeconfig() and get-password()
  export CONTEXT="centos"
  export KUBE_SERVER="http://${MASTER_ADVERTISE_ADDRESS}:8080"
  source "${KUBE_ROOT}/cluster/common.sh"

  # set kubernetes user and password
  get-password
  create-kubeconfig
}

# Delete a kubernetes cluster
function kube-down() {
  for master in ${MASTERS}; do
    tear-down-master ${master}
  done

  for node in ${NODES}; do
    tear-down-node ${node}
  done
}

function troubleshoot-master() {
  # Troubleshooting on master if all required daemons are active.
  echo "[INFO] Troubleshooting on master $1"
  local -a required_daemon=("kube-apiserver" "kube-controller-manager" "kube-scheduler")
  local daemon
  local daemon_status
  printf "%-24s %-10s \n" "PROCESS" "STATUS"
  for daemon in "${required_daemon[@]}"; do
    local rc=0
    kube-ssh "${1}" "sudo systemctl is-active ${daemon}" >/dev/null 2>&1 || rc="$?"
    if [[ "${rc}" -ne "0" ]]; then
      daemon_status="inactive"
    else
      daemon_status="active"
    fi
    printf "%-24s %s\n" ${daemon} ${daemon_status}
  done
  printf "\n"
}

function troubleshoot-node() {
  # Troubleshooting on node if all required daemons are active.
  echo "[INFO] Troubleshooting on node ${1}"
  local -a required_daemon=("kube-proxy" "kubelet" "docker" "flannel")
  local daemon
  local daemon_status
  printf "%-24s %-10s \n" "PROCESS" "STATUS"
  for daemon in "${required_daemon[@]}"; do
    local rc=0
    kube-ssh "${1}" "sudo systemctl is-active ${daemon}" >/dev/null 2>&1 || rc="$?"
    if [[ "${rc}" -ne "0" ]]; then
      daemon_status="inactive"
    else
      daemon_status="active"
    fi
    printf "%-24s %s\n" ${daemon} ${daemon_status}
  done
  printf "\n"
}

# Clean up on master
function tear-down-master() {
echo "[INFO] tear-down-master on $1"
  for service_name in etcd kube-apiserver kube-controller-manager kube-scheduler ; do
      service_file="/usr/lib/systemd/system/${service_name}.service"
      kube-ssh "$1" " \
        if [[ -f $service_file ]]; then \
          sudo systemctl stop $service_name; \
          sudo systemctl disable $service_name; \
          sudo rm -f $service_file; \
        fi"
  done
  kube-ssh "${1}" "sudo rm -rf /opt/kubernetes"
  kube-ssh "${1}" "sudo rm -rf ${KUBE_TEMP}"
  kube-ssh "${1}" "sudo rm -rf /var/lib/etcd"
}

# Clean up on node
function tear-down-node() {
echo "[INFO] tear-down-node on $1"
  for service_name in kube-proxy kubelet docker flannel ; do
      service_file="/usr/lib/systemd/system/${service_name}.service"
      kube-ssh "$1" " \
        if [[ -f $service_file ]]; then \
          sudo systemctl stop $service_name; \
          sudo systemctl disable $service_name; \
          sudo rm -f $service_file; \
        fi"
  done
  kube-ssh "$1" "sudo rm -rf /run/flannel"
  kube-ssh "$1" "sudo rm -rf /opt/kubernetes"
  kube-ssh "$1" "sudo rm -rf ${KUBE_TEMP}"
}

# Generate the CA certificates for k8s components
function make-ca-cert() {
  echo "[INFO] make-ca-cert"
  bash "${ROOT}/../saltbase/salt/generate-cert/make-ca-cert.sh" "${MASTER_ADVERTISE_IP}" "IP:${MASTER_ADVERTISE_IP},IP:${SERVICE_CLUSTER_IP_RANGE%.*}.1,DNS:kubernetes,DNS:kubernetes.default,DNS:kubernetes.default.svc,DNS:kubernetes.default.svc.cluster.local"
}

# Provision master
#
# Assumed vars:
#   $1 (master)
#   KUBE_TEMP
#   ETCD_SERVERS
#   ETCD_INITIAL_CLUSTER
#   SERVICE_CLUSTER_IP_RANGE
#   MASTER_ADVERTISE_ADDRESS
function provision-master() {
  echo "[INFO] Provision master on $1"
  local master="$1"
  local master_ip="${master#*@}"
  local etcd_name="$2"
  ensure-setup-dir "${master}"

  kube-scp "${master}" "${ROOT}/ca-cert ${ROOT}/binaries/master ${ROOT}/master ${ROOT}/config-default.sh ${ROOT}/util.sh" "${KUBE_TEMP}"
  kube-ssh "${master}" " \
    sudo rm -rf /opt/kubernetes/bin; \
    sudo cp -r ${KUBE_TEMP}/master/bin /opt/kubernetes; \
    sudo mkdir -p /srv/kubernetes; sudo cp -f ${KUBE_TEMP}/ca-cert/* /srv/kubernetes; \
    sudo chmod -R +x /opt/kubernetes/bin; \
    sudo ln -sf /opt/kubernetes/bin/* /usr/local/bin/; \
    sudo bash ${KUBE_TEMP}/master/scripts/etcd.sh ${etcd_name} ${master_ip} ${ETCD_INITIAL_CLUSTER}; \
    sudo bash ${KUBE_TEMP}/master/scripts/apiserver.sh ${master_ip} ${ETCD_SERVERS} ${SERVICE_CLUSTER_IP_RANGE} ${ADMISSION_CONTROL}; \
    sudo bash ${KUBE_TEMP}/master/scripts/controller-manager.sh ${MASTER_ADVERTISE_ADDRESS}; \
    sudo bash ${KUBE_TEMP}/master/scripts/scheduler.sh ${MASTER_ADVERTISE_ADDRESS}"
}

# Post-provision master, run after all masters were provisioned
#
# Assumed vars:
#   $1 (master)
#   KUBE_TEMP
#   ETCD_SERVERS
#   FLANNEL_NET
function post-provision-master() {
  echo "[INFO] Post provision master on $1"
  local master=$1
  kube-ssh "${master}" " \
    sudo bash ${KUBE_TEMP}/master/scripts/flannel.sh ${ETCD_SERVERS} ${FLANNEL_NET}; \
    sudo bash ${KUBE_TEMP}/master/scripts/post-etcd.sh"
}

# Provision node
#
# Assumed vars:
#   $1 (node)
#   KUBE_TEMP
#   ETCD_SERVERS
#   FLANNEL_NET
#   MASTER_ADVERTISE_ADDRESS
#   DOCKER_OPTS
#   DNS_SERVER_IP
#   DNS_DOMAIN
function provision-node() {
  echo "[INFO] Provision node on $1"
  local node=$1
  local node_ip=${node#*@}
  local dns_ip=${DNS_SERVER_IP#*@}
  local dns_domain=${DNS_DOMAIN#*@}
  ensure-setup-dir ${node}

  kube-scp ${node} "${ROOT}/binaries/node ${ROOT}/node ${ROOT}/config-default.sh ${ROOT}/util.sh" ${KUBE_TEMP}
  kube-ssh "${node}" " \
    rm -rf /opt/kubernetes/bin; \
    sudo cp -r ${KUBE_TEMP}/node/bin /opt/kubernetes; \
    sudo chmod -R +x /opt/kubernetes/bin; \
    sudo ln -s /opt/kubernetes/bin/* /usr/local/bin/; \
    sudo bash ${KUBE_TEMP}/node/scripts/flannel.sh ${ETCD_SERVERS} ${FLANNEL_NET}; \
    sudo bash ${KUBE_TEMP}/node/scripts/docker.sh \"${DOCKER_OPTS}\"; \
    sudo bash ${KUBE_TEMP}/node/scripts/kubelet.sh ${MASTER_ADVERTISE_ADDRESS} ${node_ip} ${dns_ip} ${dns_domain}; \
    sudo bash ${KUBE_TEMP}/node/scripts/proxy.sh ${MASTER_ADVERTISE_ADDRESS}"
}

# Create dirs that'll be used during setup on target machine.
#
# Assumed vars:
#   KUBE_TEMP
function ensure-setup-dir() {
  kube-ssh "${1}" "mkdir -p ${KUBE_TEMP}; \
                   sudo mkdir -p /opt/kubernetes/bin; \
                   sudo mkdir -p /opt/kubernetes/cfg"
}

# Run command over ssh
function kube-ssh() {
  local host="$1"
  shift
  ssh ${SSH_OPTS} -t "${host}" "$@" >/dev/null 2>&1
}

# Copy file recursively over ssh
function kube-scp() {
  local host="$1"
  local src=($2)
  local dst="$3"
  scp -r ${SSH_OPTS} ${src[*]} "${host}:${dst}"
}

# Ensure that we have a password created for validating to the master. Will
# read from kubeconfig if available.
#
# Vars set:
#   KUBE_USER
#   KUBE_PASSWORD
function get-password {
  load-or-gen-kube-basicauth
  if [[ -z "${KUBE_USER}" || -z "${KUBE_PASSWORD}" ]]; then
    KUBE_USER=admin
    KUBE_PASSWORD=$(python -c 'import string,random; \
      print("".join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(16)))')
  fi
}
