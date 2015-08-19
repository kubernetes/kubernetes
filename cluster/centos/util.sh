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

# A library of helper functions that each provider hosting Kubernetes must implement to use cluster/kube-*.sh scripts.

# exit on any error
set -e

SSH_OPTS="-oStrictHostKeyChecking=no -oUserKnownHostsFile=/dev/null -oLogLevel=ERROR"

# Use the config file specified in $KUBE_CONFIG_FILE, or default to
# config-default.sh.
KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
readonly ROOT=$(dirname "${BASH_SOURCE}")
source "${ROOT}/${KUBE_CONFIG_FILE:-"config-default.sh"}"
source "$KUBE_ROOT/cluster/common.sh"


KUBECTL_PATH=${KUBE_ROOT}/cluster/centos/binaries/kubectl

# Directory to be used for master and minion provisioning.
KUBE_TEMP="~/kubernetes"


# Must ensure that the following ENV vars are set
function detect-master() {
  KUBE_MASTER=$MASTER
  KUBE_MASTER_IP=${MASTER#*@}
  echo "KUBE_MASTER_IP: ${KUBE_MASTER_IP}" 1>&2
  echo "KUBE_MASTER: ${MASTER}" 1>&2
}

# Get minion IP addresses and store in KUBE_MINION_IP_ADDRESSES[]
function detect-minions() {
  KUBE_MINION_IP_ADDRESSES=()
  for minion in ${MINIONS}; do
    KUBE_MINION_IP_ADDRESSES+=("${minion#*@}")
  done
  echo "KUBE_MINION_IP_ADDRESSES: [${KUBE_MINION_IP_ADDRESSES[*]}]" 1>&2
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
  "${KUBE_ROOT}/cluster/validate-cluster.sh"
}

# Instantiate a kubernetes cluster
function kube-up() {
  provision-master

  for minion in ${MINIONS}; do
    provision-minion ${minion}
  done

  verify-master
  for minion in ${MINIONS}; do
    verify-minion ${minion}
  done

  detect-master

  # set CONTEXT and KUBE_SERVER values for create-kubeconfig() and get-password()
  export CONTEXT="centos"
  export KUBE_SERVER="http://${KUBE_MASTER_IP}:8080"
  source "${KUBE_ROOT}/cluster/common.sh"

  # set kubernetes user and password
  get-password
  create-kubeconfig
}

# Delete a kubernetes cluster
function kube-down() {
  tear-down-master
  for minion in ${MINIONS}; do
    tear-down-minion ${minion}
  done
}


function verify-master() {
  # verify master has all required daemons
  printf "[INFO] Validating master ${MASTER}"
  local -a required_daemon=("kube-apiserver" "kube-controller-manager" "kube-scheduler")
  local validated="1"
  local try_count=0
  until [[ "$validated" == "0" ]]; do
    validated="0"
    local daemon
    for daemon in "${required_daemon[@]}"; do
      local rc=0
      kube-ssh "${MASTER}" "pgrep -f ${daemon}" >/dev/null 2>&1 || rc="$?"
      if [[ "${rc}" -ne "0" ]]; then
        printf "."
        validated="1"
        ((try_count=try_count+2))
        if [[ ${try_count} -gt ${PROCESS_CHECK_TIMEOUT} ]]; then
          printf "\nWarning: Process \"${daemon}\" status check timeout, please check manually.\n"
          exit 1
        fi
        sleep 2
      fi
    done
  done
  printf "\n"

}

function verify-minion() {
  # verify minion has all required daemons
  printf "[INFO] Validating minion ${1}"
  local -a required_daemon=("kube-proxy" "kubelet" "docker")
  local validated="1"
  local try_count=0
  until [[ "$validated" == "0" ]]; do
    validated="0"
    local daemon
    for daemon in "${required_daemon[@]}"; do
      local rc=0
      kube-ssh "${1}" "pgrep -f ${daemon}" >/dev/null 2>&1 || rc="$?"
      if [[ "${rc}" -ne "0" ]]; then
        printf "."
        validated="1"
        ((try_count=try_count+2))
        if [[ ${try_count} -gt ${PROCESS_CHECK_TIMEOUT} ]] ; then
          printf "\nWarning: Process \"${daemon}\" status check timeout, please check manually.\n"
          exit 1
        fi
        sleep 2
      fi
    done
  done
  printf "\n"
}

# Clean up on master
function tear-down-master() {
echo "[INFO] tear-down-master on ${MASTER}"
  for service_name in etcd kube-apiserver kube-controller-manager kube-scheduler ; do
      service_file="/usr/lib/systemd/system/${service_name}.service"
      (
        echo "if [[ -f $service_file ]]; then"
        echo "systemctl stop $service_name"
        echo "systemctl disable $service_name"
        echo "rm -f $service_file"
        echo "fi"
      ) | kube-ssh "$MASTER"
  done
  kube-ssh "${MASTER}" "rm -rf /opt/kubernetes"
  kube-ssh "${MASTER}" "rm -rf ${KUBE_TEMP}"
  kube-ssh "${MASTER}" "rm -rf /var/lib/etcd"
}

# Clean up on minion
function tear-down-minion() {
echo "[INFO] tear-down-minion on $1"
  for service_name in kube-proxy kubelet docker flannel ; do
      service_file="/usr/lib/systemd/system/${service_name}.service"
      (
        echo "if [[ -f $service_file ]]; then"
        echo "systemctl stop $service_name"
        echo "systemctl disable $service_name"
        echo "rm -f $service_file"
        echo "fi"
      ) | kube-ssh "$1"
  done
  kube-ssh "$1" "rm -rf /run/flannel"
  kube-ssh "$1" "rm -rf /opt/kubernetes"
  kube-ssh "$1" "rm -rf ${KUBE_TEMP}"
}

# Provision master
#
# Assumed vars:
#   MASTER
#   KUBE_TEMP
#   ETCD_SERVERS
#   SERVICE_CLUSTER_IP_RANGE
function provision-master() {
  echo "[INFO] Provision master on ${MASTER}"
  local master_ip=${MASTER#*@}
  ensure-setup-dir ${MASTER}

  # scp -r ${SSH_OPTS} master config-default.sh copy-files.sh util.sh "${MASTER}:${KUBE_TEMP}" 
  kube-scp ${MASTER} "${ROOT}/binaries/master ${ROOT}/master ${ROOT}/config-default.sh ${ROOT}/util.sh" "${KUBE_TEMP}" 
  (
    echo "cp -r ${KUBE_TEMP}/master/bin /opt/kubernetes"
    echo "chmod -R +x /opt/kubernetes/bin"

    echo "bash ${KUBE_TEMP}/master/scripts/etcd.sh"
    echo "bash ${KUBE_TEMP}/master/scripts/apiserver.sh ${master_ip} ${ETCD_SERVERS} ${SERVICE_CLUSTER_IP_RANGE}"
    echo "bash ${KUBE_TEMP}/master/scripts/controller-manager.sh ${master_ip}"
    echo "bash ${KUBE_TEMP}/master/scripts/scheduler.sh ${master_ip}"

  ) | kube-ssh "${MASTER}"
}


# Provision minion
#
# Assumed vars:
#   $1 (minion)
#   MASTER
#   KUBE_TEMP
#   ETCD_SERVERS
#   FLANNEL_NET
#   DOCKER_OPTS
function provision-minion() {
  echo "[INFO] Provision minion on $1"
  local master_ip=${MASTER#*@}
  local minion=$1
  local minion_ip=${minion#*@}
  ensure-setup-dir ${minion_ip}

  # scp -r ${SSH_OPTS} minion config-default.sh copy-files.sh util.sh "${minion_ip}:${KUBE_TEMP}" 
  kube-scp ${minion_ip} "${ROOT}/binaries/minion ${ROOT}/minion ${ROOT}/config-default.sh ${ROOT}/util.sh" ${KUBE_TEMP}
  (
    echo "cp -r ${KUBE_TEMP}/minion/bin /opt/kubernetes"
    echo "chmod -R +x /opt/kubernetes/bin"

    echo "bash ${KUBE_TEMP}/minion/scripts/flannel.sh ${ETCD_SERVERS} ${FLANNEL_NET}"
    echo "bash ${KUBE_TEMP}/minion/scripts/docker.sh ${DOCKER_OPTS}"
    echo "bash ${KUBE_TEMP}/minion/scripts/kubelet.sh ${master_ip} ${minion_ip}"
    echo "bash ${KUBE_TEMP}/minion/scripts/proxy.sh ${master_ip}"

  ) | kube-ssh "${minion_ip}"
}

# Create dirs that'll be used during setup on target machine.
#
# Assumed vars:
#   KUBE_TEMP
function ensure-setup-dir() {
  (
    echo "mkdir -p ${KUBE_TEMP}"
    echo "mkdir -p /opt/kubernetes/bin"
    echo "mkdir -p /opt/kubernetes/cfg"
  ) | kube-ssh "${1}"
}

# Run command over ssh
function kube-ssh() {
  local host="$1"
  shift
  ssh ${SSH_OPTS-} "${host}" "$@" >/dev/null 2>&1
}

# Copy file recursively over ssh
function kube-scp() {
  local host="$1"
  local src=($2)
  local dst="$3"
  scp -r ${SSH_OPTS-} ${src[*]} "${host}:${dst}"
}

# Ensure that we have a password created for validating to the master. Will
# read from kubeconfig if available.
#
# Vars set:
#   KUBE_USER
#   KUBE_PASSWORD
function get-password {
  get-kubeconfig-basicauth
  if [[ -z "${KUBE_USER}" || -z "${KUBE_PASSWORD}" ]]; then
    KUBE_USER=admin
    KUBE_PASSWORD=$(python -c 'import string,random; \
      print "".join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(16))')
  fi
}
