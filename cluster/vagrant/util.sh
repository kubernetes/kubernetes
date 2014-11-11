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
  echo "KUBE_MASTER_IP: ${KUBE_MASTER_IP}"
}

# Get minion IP addresses and store in KUBE_MINION_IP_ADDRESSES[]
function detect-minions {
  echo "Minions already detected"
}

# Verify prereqs on host machine
function verify-prereqs {
  for x in vagrant virtualbox; do
    if ! which "$x" >/dev/null; then
      echo "Can't find $x in PATH, please fix and retry."
      exit 1
    fi
  done
}

# Instantiate a kubernetes cluster
function kube-up {
  get-password
  vagrant up

  echo "Each machine instance has been created."
  echo "  Now waiting for the Salt provisioning process to complete on each machine."
  echo "  This can take some time based on your network, disk, and cpu speed."
  echo "  It is possible for an error to occur during Salt provision of cluster and this could loop forever."

  # verify master has all required daemons
  echo "Validating master"
  local machine="master"
  local -a required_daemon=("salt-master" "salt-minion" "apiserver" "nginx" "controller-manager" "scheduler")
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
  for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
    local machine="${MINION_NAMES[$i]}"
    local count="0"
    until [[ "$count" == "1" ]]; do
      local minions
      minions=$("${KUBE_ROOT}/cluster/kubecfg.sh" -template '{{range.items}}{{.id}}:{{end}}' list minions)
      count=$(echo $minions | grep -c "${MINION_NAMES[i]}") || {
        printf "."
        sleep 2
        count="0"
      }
    done
  done

  echo
  echo "Kubernetes cluster created."
  echo
  echo "Kubernetes cluster is running.  The master is running at:"
  echo
  echo "  https://${KUBE_MASTER_IP}"
  echo
  echo "The user name and password to use is located in ~/.kubernetes_auth."
  echo
}

# Delete a kubernetes cluster
function kube-down {
  vagrant destroy -f
}

# Update a kubernetes cluster with latest source
function kube-push {
  vagrant provision
}

# Execute prior to running tests to build a release if required for env
function test-build-release {
  # Make a release
  "${KUBE_ROOT}/build/release.sh"
}

# Execute prior to running tests to initialize required structure
function test-setup {
  echo "Vagrant test setup complete"
}

# Execute after running tests to perform any required clean-up
function test-teardown {
  echo "Vagrant ignores tear-down"
}

# Set the {user} and {password} environment values required to interact with provider
function get-password {
  export KUBE_USER=vagrant
  export KUBE_PASSWORD=vagrant
  echo "Using credentials: $KUBE_USER:$KUBE_PASSWORD"
}

# Find the minion name based on the IP address
function find-minion-by-ip {
  local ip="$1"
  local ip_pattern="${MINION_IP_BASE}(.*)"

  # This is subtle.  We map 10.245.2.2 -> minion-1.  We do this by matching a
  # regexp and using the capture to construct the name.
  [[ $ip =~ $ip_pattern ]] || {
    return 1
  }

  echo "minion-$((${BASH_REMATCH[1]} - 1))"
}

# SSH to a node by name ($1) and run a command ($2).
function ssh-to-node {
  local node="$1"
  local cmd="$2"
  local machine
  machine=$(find-minion-by-ip $node)
  vagrant ssh "${machine}" -c "${cmd}" | grep -v "Connection to.*closed"
}

# Restart the kube-proxy on a node ($1)
function restart-kube-proxy {
  ssh-to-node "$1" "sudo systemctl restart kube-proxy"
}

function setup-monitoring {
    echo "TODO"
}

# Perform preparations required to run e2e tests
function prepare-e2e() {
  echo "Vagrant doesn't need special preparations for e2e tests"
}
