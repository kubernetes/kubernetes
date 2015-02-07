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

# A library of helper functions that each provider hosting Kubernetes must implement to use cluster/kube-*.sh scripts.

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
source "${KUBE_ROOT}/cluster/vagrant/${KUBE_CONFIG_FILE-"config-default.sh"}"

function detect-master () {
  KUBE_MASTER_IP=$MASTER_IP
  echo "KUBE_MASTER_IP: ${KUBE_MASTER_IP}" 1>&2
}

# Get minion IP addresses and store in KUBE_MINION_IP_ADDRESSES[]
function detect-minions {
  echo "Minions already detected" 1>&2
  KUBE_MINION_IP_ADDRESSES=("${MINION_IPS[@]}")
}

# Verify prereqs on host machine  Also sets exports USING_KUBE_SCRIPTS=true so
# that our Vagrantfile doesn't error out.
function verify-prereqs {
  for x in vagrant virtualbox; do
    if ! which "$x" >/dev/null; then
      echo "Can't find $x in PATH, please fix and retry."
      exit 1
    fi
  done

  # Set VAGRANT_CWD to KUBE_ROOT so that we find the right Vagrantfile no
  # matter what directory the tools are called from.
  export VAGRANT_CWD="${KUBE_ROOT}"

  export USING_KUBE_SCRIPTS=true
}

# Create a temp dir that'll be deleted at the end of this bash session.
#
# Vars set:
#   KUBE_TEMP
function ensure-temp-dir {
  if [[ -z ${KUBE_TEMP-} ]]; then
    export KUBE_TEMP=$(mktemp -d -t kubernetes.XXXXXX)
    trap 'rm -rf "${KUBE_TEMP}"' EXIT
  fi
}

# Create a set of provision scripts for the master and each of the minions
function create-provision-scripts {
  ensure-temp-dir

  (
    echo "#! /bin/bash"
    echo "KUBE_ROOT=/vagrant"
    echo "MASTER_NAME='${INSTANCE_PREFIX}-master'"
    echo "MASTER_IP='${MASTER_IP}'"
    echo "MINION_NAMES=(${MINION_NAMES[@]})"
    echo "MINION_IPS=(${MINION_IPS[@]})"
    echo "NODE_IP='${MASTER_IP}'"
    echo "CONTAINER_SUBNET='${CONTAINER_SUBNET}'"
    echo "CONTAINER_NETMASK='${MASTER_CONTAINER_NETMASK}'"
    echo "MASTER_CONTAINER_SUBNET='${MASTER_CONTAINER_SUBNET}'"
    echo "CONTAINER_ADDR='${MASTER_CONTAINER_ADDR}'"
    echo "MINION_CONTAINER_NETMASKS='${MINION_CONTAINER_NETMASKS[@]}'"
    echo "MINION_CONTAINER_SUBNETS=(${MINION_CONTAINER_SUBNETS[@]})"
    echo "PORTAL_NET='${PORTAL_NET}'"
    echo "MASTER_USER='${MASTER_USER}'"
    echo "MASTER_PASSWD='${MASTER_PASSWD}'"
    echo "ENABLE_NODE_MONITORING='${ENABLE_NODE_MONITORING:-false}'"
    echo "ENABLE_NODE_LOGGING='${ENABLE_NODE_LOGGING:-false}'"
    echo "LOGGING_DESTINATION='${LOGGING_DESTINATION:-}'"
    echo "ENABLE_CLUSTER_DNS='${ENABLE_CLUSTER_DNS:-false}'"
    echo "DNS_SERVER_IP='${DNS_SERVER_IP:-}'"
    echo "DNS_DOMAIN='${DNS_DOMAIN:-}'"
    echo "RUNTIME_CONFIG='${RUNTIME_CONFIG:-}'"
    grep -v "^#" "${KUBE_ROOT}/cluster/vagrant/provision-master.sh"
    grep -v "^#" "${KUBE_ROOT}/cluster/vagrant/provision-network.sh"
  ) > "${KUBE_TEMP}/master-start.sh"

  for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
    (
      echo "#! /bin/bash"
      echo "MASTER_NAME='${MASTER_NAME}'"
      echo "MASTER_IP='${MASTER_IP}'"
      echo "MINION_NAMES=(${MINION_NAMES[@]})"
      echo "MINION_IPS=(${MINION_IPS[@]})"
      echo "MINION_IP='${MINION_IPS[$i]}'"
      echo "MINION_ID='$i'"
      echo "NODE_IP='${MINION_IPS[$i]}'"
      echo "MASTER_CONTAINER_SUBNET='${MASTER_CONTAINER_SUBNET}'"
      echo "CONTAINER_ADDR='${MINION_CONTAINER_ADDRS[$i]}'"
      echo "CONTAINER_NETMASK='${MINION_CONTAINER_NETMASKS[$i]}'"
      echo "MINION_CONTAINER_SUBNETS=(${MINION_CONTAINER_SUBNETS[@]})"
      echo "CONTAINER_SUBNET='${CONTAINER_SUBNET}'"
      echo "DOCKER_OPTS='${EXTRA_DOCKER_OPTS-}'"
      grep -v "^#" "${KUBE_ROOT}/cluster/vagrant/provision-minion.sh"
      grep -v "^#" "${KUBE_ROOT}/cluster/vagrant/provision-network.sh"
    ) > "${KUBE_TEMP}/minion-start-${i}.sh"
  done
}

function verify-cluster {
  echo "Each machine instance has been created/updated."
  echo "  Now waiting for the Salt provisioning process to complete on each machine."
  echo "  This can take some time based on your network, disk, and cpu speed."
  echo "  It is possible for an error to occur during Salt provision of cluster and this could loop forever."

  # verify master has all required daemons
  echo "Validating master"
  local machine="master"
  local -a required_daemon=("salt-master" "salt-minion" "kube-apiserver" "nginx" "kube-controller-manager" "kube-scheduler")
  local validated="1"
  until [[ "$validated" == "0" ]]; do
    validated="0"
    local daemon
    for daemon in "${required_daemon[@]}"; do
      vagrant ssh "$machine" -c "which '${daemon}'" >/dev/null 2>&1 || {
        printf "."
        validated="1"
        sleep 2
      }
    done
  done

  # verify each minion has all required daemons
  local i
  for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
    echo "Validating ${VAGRANT_MINION_NAMES[$i]}"
    local machine=${VAGRANT_MINION_NAMES[$i]}
    local -a required_daemon=("salt-minion" "kubelet" "docker")
    local validated="1"
    until [[ "$validated" == "0" ]]; do
      validated="0"
      local daemon
      for daemon in "${required_daemon[@]}"; do
        vagrant ssh "$machine" -c "which $daemon" >/dev/null 2>&1 || {
          printf "."
          validated="1"
          sleep 2
        }
      done
    done
  done

  echo
  echo "Waiting for each minion to be registered with cloud provider"
  for (( i=0; i<${#MINION_IPS[@]}; i++)); do
    local machine="${MINION_IPS[$i]}"
    local count="0"
    until [[ "$count" == "1" ]]; do
      local minions
      minions=$("${KUBE_ROOT}/cluster/kubectl.sh" get minions -o template -t '{{range.items}}{{.id}}:{{end}}')
      count=$(echo $minions | grep -c "${MINION_IPS[i]}") || {
        printf "."
        sleep 2
        count="0"
      }
    done
  done

  (
    echo
    echo "Kubernetes cluster is running.  The master is running at:"
    echo
    echo "  https://${MASTER_IP}"
    echo
    echo "The user name and password to use is located in ~/.kubernetes_vagrant_auth."
    echo
    )
}


# Instantiate a kubernetes cluster
function kube-up {
  get-password
  create-provision-scripts

  vagrant up

  local kube_cert=".kubecfg.vagrant.crt"
  local kube_key=".kubecfg.vagrant.key"
  local ca_cert=".kubernetes.vagrant.ca.crt"

  (umask 077
   vagrant ssh master -- sudo cat /srv/kubernetes/kubecfg.crt >"${HOME}/${kube_cert}" 2>/dev/null
   vagrant ssh master -- sudo cat /srv/kubernetes/kubecfg.key >"${HOME}/${kube_key}" 2>/dev/null
   vagrant ssh master -- sudo cat /srv/kubernetes/ca.crt >"${HOME}/${ca_cert}" 2>/dev/null

   cat <<EOF >"${HOME}/.kubernetes_vagrant_auth"
{
  "User": "$KUBE_USER",
  "Password": "$KUBE_PASSWORD",
  "CAFile": "$HOME/$ca_cert",
  "CertFile": "$HOME/$kube_cert",
  "KeyFile": "$HOME/$kube_key"
}
EOF

   chmod 0600 ~/.kubernetes_vagrant_auth "${HOME}/${kube_cert}" \
     "${HOME}/${kube_key}" "${HOME}/${ca_cert}"
  )

  verify-cluster
}

# Delete a kubernetes cluster
function kube-down {
  vagrant destroy -f
}

# Update a kubernetes cluster with latest source
function kube-push {
  get-password
  create-provision-scripts
  vagrant provision
}

# Execute prior to running tests to build a release if required for env
function test-build-release {
  # Make a release
  "${KUBE_ROOT}/build/release.sh"
}

# Execute prior to running tests to initialize required structure
function test-setup {
  echo "Vagrant test setup complete" 1>&2
}

# Execute after running tests to perform any required clean-up
function test-teardown {
  echo "Vagrant ignores tear-down" 1>&2
}

# Set the {user} and {password} environment values required to interact with provider
function get-password {
  export KUBE_USER=vagrant
  export KUBE_PASSWORD=vagrant
  echo "Using credentials: $KUBE_USER:$KUBE_PASSWORD" 1>&2
}

# Find the minion name based on the IP address
function find-vagrant-name-by-ip {
  local ip="$1"
  local ip_pattern="${MINION_IP_BASE}(.*)"

  # This is subtle.  We map 10.245.2.2 -> minion-1.  We do this by matching a
  # regexp and using the capture to construct the name.
  [[ $ip =~ $ip_pattern ]] || {
    return 1
  }

  echo "minion-$((${BASH_REMATCH[1]} - 1))"
}

# Find the vagrant machien name based on the host name of the minion
function find-vagrant-name-by-minion-name {
  local ip="$1"
  if [[ "$ip" == "${INSTANCE_PREFIX}-master" ]]; then
    echo "master"
    return $?
  fi
  local ip_pattern="${INSTANCE_PREFIX}-minion-(.*)"

  [[ $ip =~ $ip_pattern ]] || {
    return 1
  }

  echo "minion-${BASH_REMATCH[1]}"
}


# SSH to a node by name or IP ($1) and run a command ($2).
function ssh-to-node {
  local node="$1"
  local cmd="$2"
  local machine

  machine=$(find-vagrant-name-by-ip $node) || true
  [[ -n ${machine-} ]] || machine=$(find-vagrant-name-by-minion-name $node) || true
  [[ -n ${machine-} ]] || {
    echo "Cannot find machine to ssh to: $1"
    return 1
  }

  vagrant ssh "${machine}" -c "${cmd}"
}

# Restart the kube-proxy on a node ($1)
function restart-kube-proxy {
  ssh-to-node "$1" "sudo systemctl restart kube-proxy"
}

# Restart the apiserver
function restart-apiserver {
  ssh-to-node "${master}" "sudo systemctl restart kube-apiserver"
}

function setup-monitoring-firewall {
  echo "TODO" 1>&2
}

function teardown-monitoring-firewall {
  echo "TODO" 1>&2
}

# Perform preparations required to run e2e tests
function prepare-e2e() {
  echo "Vagrant doesn't need special preparations for e2e tests" 1>&2
}

function setup-logging-firewall {
  echo "TODO: setup logging"
}

function teardown-logging-firewall {
  echo "TODO: teardown logging"
}
