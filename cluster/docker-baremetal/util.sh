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

# Implementation of baremetal docker based kubernetes provider
set -e

# Command used for remotely deploy
SSH_TO_NODE="ssh $SSH_OPTS -t"
SCP_TO_NODE="scp $SSH_OPTS -r"

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

# Verify ssh is available
function verify-prereqs-baremetal {
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
    # Try adding one of the default identities, with or without pass phrase.
    ssh-add || true
  fi
  # Expect at least one identity to be available.
  if ! ssh-add -L 1> /dev/null 2> /dev/null; then
    echo "Could not find or add an SSH identity."
    echo "Please start ssh-agent, add your identity, and retry."
    exit 1
  fi
}

# Deploy master (or master & node)
# 1. Copy files
# 2. Run master scripts
#
# Assumed vars:
#   MASTER
#   SSH_OPTS
function deploy-node-master-baremetal() {
  local files="${KUBE_ROOT}/cluster/images/hyperkube/master-multi.json \
  ${KUBE_ROOT}/cluster/docker"
  local dest_dir="${MASTER}:~"

  $SCP_TO_NODE $files $dest_dir
  
  local machine=$MASTER
  local cmd="sudo bash ~/docker/kube-deploy/master.sh;"

  # Remotely login to $MASTER and use $cmd to deploy k8s master
  $SSH_TO_NODE $machine $cmd
}

# Deploy nodes
# 1. Copy files
# 2. Run node scripts
#
# Assumed vars:
#   node
#   SSH_OPTS
function deploy-node-baremetal() {
  local files="${KUBE_ROOT}/cluster/docker"
  local dest_dir="$node:~"

  $SCP_TO_NODE $SSH_OPTS $files $dest_dir 

  # Remotely login to $node and use $cmd to deploy k8s node
  local machine=$node
  local cmd="sudo bash ~/docker/kube-deploy/node.sh;"

  $SSH_TO_NODE $machine $cmd 
}

# Destroy k8s cluster
#
# Assumed vars:
#   NODES
#   SSH_OPTS
#   MASTER
function kube-down-baremetal() {
  for node in ${NODES}; do
  {
    echo "... Cleaning on node ${node}"
    $SSH_TO_NODE ${node} "sudo bash ~/docker/kube-deploy/destroy.sh clear_all && rm -rf ~/docker/"
  }
  done

  echo "... Cleaning on MASTER ${MASTER}"
  $SSH_TO_NODE ${MASTER} "sudo bash ~/docker/kube-deploy/destroy.sh clear_all && rm -rf ~/docker/" 
}

# Verify cluster
# 
# Assumed vars:
#   MASTER
#   NODES
#   SSH_OPTS
function validate-cluster-baremetal() {
  # Validate master
  echo "... Validating Master $MASTER"
  $SSH_TO_NODE $MASTER "bash ~/docker/kube-deploy/verify.sh master"

  # Validate nodes
  for node in $NODES
  do
    {
      if [ "$node" != $MASTER ]; then
        echo "... Validating Node $node"
        $SSH_TO_NODE $node "bash ~/docker/kube-deploy/verify.sh node"
      fi
    }
  done
}