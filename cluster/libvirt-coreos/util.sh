#!/bin/bash

# Copyright 2014 The Kubernetes Authors All rights reserved.
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
readonly ROOT=$(dirname "${BASH_SOURCE}")
source "$ROOT/${KUBE_CONFIG_FILE:-"config-default.sh"}"
source "$KUBE_ROOT/cluster/common.sh"

export LIBVIRT_DEFAULT_URI=qemu:///system

readonly POOL=kubernetes
readonly POOL_PATH="$(cd $ROOT && pwd)/libvirt_storage_pool"

# join <delim> <list...>
# Concatenates the list elements with the delimiter passed as first parameter
#
# Ex: join , a b c
#  -> a,b,c
function join {
  local IFS="$1"
  shift
  echo "$*"
}

# Must ensure that the following ENV vars are set
function detect-master {
  KUBE_MASTER_IP=$MASTER_IP
  KUBE_MASTER=$MASTER_NAME
  export KUBERNETES_MASTER=http://$KUBE_MASTER_IP:8080
  echo "KUBE_MASTER_IP: $KUBE_MASTER_IP"
  echo "KUBE_MASTER: $KUBE_MASTER"
}

# Get minion IP addresses and store in KUBE_MINION_IP_ADDRESSES[]
function detect-minions {
  KUBE_MINION_IP_ADDRESSES=("${MINION_IPS[@]}")
}

# Verify prereqs on host machine
function verify-prereqs {
  if ! which virsh >/dev/null; then
      echo "Can't find virsh in PATH, please fix and retry." >&2
      exit 1
  fi
  if ! virsh nodeinfo >/dev/null; then
      exit 1
  fi
  if [[ "$(</sys/kernel/mm/ksm/run)" -ne "1" ]]; then
      echo "KSM is not enabled" >&2
      echo "Enabling it would reduce the memory footprint of large clusters" >&2
      if [[ -t 0 ]]; then
          read -t 5 -n 1 -p "Do you want to enable KSM (requires root password) (y/n)? " answer
          echo ""
          if [[ "$answer" == 'y' ]]; then
              su -c 'echo 1 > /sys/kernel/mm/ksm/run'
          fi
      else
        echo "You can enable it with (as root):" >&2
        echo "" >&2
        echo "  echo 1 > /sys/kernel/mm/ksm/run" >&2
        echo "" >&2
      fi
  fi
}

# Destroy the libvirt storage pool and all the images inside
#
# If 'keep_base_image' is passed as first parameter,
# the base image is kept, as well as the storage pool.
# All the other images are deleted.
function destroy-pool {
  virsh pool-info $POOL >/dev/null 2>&1 || return

  rm -rf "$POOL_PATH"/kubernetes/*
  rm -rf "$POOL_PATH"/kubernetes_config*/*
  local vol
  virsh vol-list $POOL | awk 'NR>2 && !/^$/ && $1 ~ /^kubernetes/ {print $1}' | \
      while read vol; do
        virsh vol-delete $vol --pool $POOL
      done

  [[ "$1" == 'keep_base_image' ]] && return

  set +e
  virsh vol-delete coreos_base.img --pool $POOL
  virsh pool-destroy $POOL
  rmdir "$POOL_PATH"
  set -e
}

# Creates the libvirt storage pool and populate it with
# - the CoreOS base image
# - the kubernetes binaries
function initialize-pool {
  mkdir -p "$POOL_PATH"
  if ! virsh pool-info $POOL >/dev/null 2>&1; then
      virsh pool-create-as $POOL dir --target "$POOL_PATH"
  fi

  wget -N -P "$ROOT" http://${COREOS_CHANNEL:-alpha}.release.core-os.net/amd64-usr/current/coreos_production_qemu_image.img.bz2
  if [[ "$ROOT/coreos_production_qemu_image.img.bz2" -nt "$POOL_PATH/coreos_base.img" ]]; then
      bunzip2 -f -k "$ROOT/coreos_production_qemu_image.img.bz2"
      virsh vol-delete coreos_base.img --pool $POOL 2> /dev/null || true
      mv "$ROOT/coreos_production_qemu_image.img" "$POOL_PATH/coreos_base.img"
  fi
  # if ! virsh vol-list $POOL | grep -q coreos_base.img; then
  #     virsh vol-create-as $POOL coreos_base.img 10G --format qcow2
  #     virsh vol-upload coreos_base.img "$ROOT/coreos_production_qemu_image.img" --pool $POOL
  # fi

  mkdir -p "$POOL_PATH/kubernetes"
  kube-push-internal

  mkdir -p "$POOL_PATH/kubernetes/manifests"
  if [[ "$ENABLE_NODE_LOGGING" == "true" ]]; then
      if [[ "$LOGGING_DESTINATION" == "elasticsearch" ]]; then
          cp "$KUBE_ROOT/cluster/saltbase/salt/fluentd-es/fluentd-es.manifest" "$POOL_PATH/kubernetes/manifests"
      elif [[ "$LOGGING_DESTINATION" == "gcp" ]]; then
          cp "$KUBE_ROOT/cluster/saltbase/salt/fluentd-gcp/fluentd-gcp.manifest" "$POOL_PATH/kubernetes/manifests"
      fi
  fi

  mkdir -p "$POOL_PATH/kubernetes/addons"
  if [[ "$ENABLE_CLUSTER_DNS" == "true" ]]; then
      render-template "$ROOT/skydns-svc.yaml" > "$POOL_PATH/kubernetes/addons/skydns-svc.yaml"
      render-template "$ROOT/skydns-rc.yaml"  > "$POOL_PATH/kubernetes/addons/skydns-rc.yaml"
  fi

  virsh pool-refresh $POOL
}

function destroy-network {
  set +e
  virsh net-destroy kubernetes_global
  virsh net-destroy kubernetes_pods
  set -e
}

function initialize-network {
  virsh net-create "$ROOT/network_kubernetes_global.xml"
  virsh net-create "$ROOT/network_kubernetes_pods.xml"
}

function render-template {
  eval "echo \"$(cat $1)\""
}

function wait-cluster-readiness {
  echo "Wait for cluster readiness"
  local kubectl="${KUBE_ROOT}/cluster/kubectl.sh"

  local timeout=120
  while [[ $timeout -ne 0 ]]; do
    nb_ready_minions=$("${kubectl}" get nodes -o go-template="{{range.items}}{{range.status.conditions}}{{.type}}{{end}}:{{end}}" --api-version=v1 2>/dev/null | tr ':' '\n' | grep -c Ready || true)
    echo "Nb ready minions: $nb_ready_minions / $NUM_MINIONS"
    if [[ "$nb_ready_minions" -eq "$NUM_MINIONS" ]]; then
        return 0
    fi

    timeout=$(($timeout-1))
    sleep .5
  done

  return 1
}

# Instantiate a kubernetes cluster
function kube-up {
  detect-master
  detect-minions
  load-or-gen-kube-bearertoken
  initialize-pool keep_base_image
  initialize-network

  readonly ssh_keys="$(cat ~/.ssh/id_*.pub | sed 's/^/  - /')"
  readonly kubernetes_dir="$POOL_PATH/kubernetes"

  local i
  for (( i = 0 ; i <= $NUM_MINIONS ; i++ )); do
    if [[ $i -eq $NUM_MINIONS ]]; then
        etcd2_initial_cluster[$i]="${MASTER_NAME}=http://${MASTER_IP}:2380"
    else
        etcd2_initial_cluster[$i]="${MINION_NAMES[$i]}=http://${MINION_IPS[$i]}:2380"
    fi
  done
  etcd2_initial_cluster=$(join , "${etcd2_initial_cluster[@]}")
  readonly machines=$(join , "${KUBE_MINION_IP_ADDRESSES[@]}")

  for (( i = 0 ; i <= $NUM_MINIONS ; i++ )); do
    if [[ $i -eq $NUM_MINIONS ]]; then
        type=master
        name=$MASTER_NAME
        public_ip=$MASTER_IP
    else
      type=minion-$(printf "%02d" $i)
      name=${MINION_NAMES[$i]}
      public_ip=${MINION_IPS[$i]}
    fi
    image=$name.img
    config=kubernetes_config_$type

    virsh vol-create-as $POOL $image 10G --format qcow2 --backing-vol coreos_base.img --backing-vol-format qcow2

    mkdir -p "$POOL_PATH/$config/openstack/latest"
    render-template "$ROOT/user_data.yml" > "$POOL_PATH/$config/openstack/latest/user_data"
    virsh pool-refresh $POOL

    domain_xml=$(mktemp)
    render-template $ROOT/coreos.xml > $domain_xml
    virsh create $domain_xml
    rm $domain_xml
  done

  export KUBE_SERVER="http://192.168.10.1:8080"
  export CONTEXT="libvirt-coreos"
  create-kubeconfig

  wait-cluster-readiness

  echo "Kubernetes cluster is running. The master is running at:"
  echo
  echo "  http://${KUBE_MASTER_IP}:8080"
  echo
  echo "You can control the Kubernetes cluster with: 'cluster/kubectl.sh'"
  echo "You can connect on the master with: 'ssh core@${KUBE_MASTER_IP}'"
}

# Delete a kubernetes cluster
function kube-down {
  virsh list | awk 'NR>2 && !/^$/ && $2 ~ /^kubernetes/ {print $2}' | \
      while read dom; do
        virsh destroy $dom
      done
  destroy-pool keep_base_image
  destroy-network
}

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

# The kubernetes binaries are pushed to a host directory which is exposed to the VM
function upload-server-tars {
  tar -x -C "$POOL_PATH/kubernetes" -f "$SERVER_BINARY_TAR" kubernetes
  rm -rf "$POOL_PATH/kubernetes/bin"
  mv "$POOL_PATH/kubernetes/kubernetes/server/bin" "$POOL_PATH/kubernetes/bin"
  rm -fr "$POOL_PATH/kubernetes/kubernetes"
}

# Update a kubernetes cluster with latest source
function kube-push {
  kube-push-internal
  ssh-to-node "$MASTER_NAME" "sudo systemctl restart kube-apiserver kube-controller-manager kube-scheduler"
  for ((i=0; i < NUM_MINIONS; i++)); do
    ssh-to-node "${MINION_NAMES[$i]}" "sudo systemctl restart kubelet kube-proxy"
  done
  wait-cluster-readiness
}

function kube-push-internal {
  case "${KUBE_PUSH:-release}" in
    release)
      kube-push-release;;
    local)
      kube-push-local;;
    *)
      echo "The only known push methods are \"release\" to use the release tarball or \"local\" to use the binaries built by make. KUBE_PUSH is set \"$KUBE_PUSH\"" >&2
      return 1;;
  esac
}

function kube-push-release {
  find-release-tars
  upload-server-tars
}

function kube-push-local {
  rm -rf "$POOL_PATH/kubernetes/bin/*"
  mkdir -p "$POOL_PATH/kubernetes/bin"
  cp "${KUBE_ROOT}/_output/local/go/bin"/* "$POOL_PATH/kubernetes/bin"
}

# Execute prior to running tests to build a release if required for env
function test-build-release {
  echo "TODO"
}

# Execute prior to running tests to initialize required structure
function test-setup {
  echo "TODO"
}

# Execute after running tests to perform any required clean-up
function test-teardown {
  kube-down
}

# SSH to a node by name or IP ($1) and run a command ($2).
function ssh-to-node {
  local node="$1"
  local cmd="$2"
  local machine

  if [[ "$node" == "$MASTER_IP" ]] || [[ "$node" =~ ^"$MINION_IP_BASE" ]]; then
      machine="$node"
  elif [[ "$node" == "$MASTER_NAME" ]]; then
      machine="$MASTER_IP"
  else
    for ((i=0; i < NUM_MINIONS; i++)); do
        if [[ "$node" == "${MINION_NAMES[$i]}" ]]; then
            machine="${MINION_IPS[$i]}"
            break
        fi
    done
  fi
  if [[ -z "$machine" ]]; then
      echo "$node is an unknown machine to ssh to" >&2
  fi
  ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ControlMaster=no "core@$machine" "$cmd"
}

# Restart the kube-proxy on a node ($1)
function restart-kube-proxy {
  ssh-to-node "$1" "sudo systemctl restart kube-proxy"
}

# Restart the apiserver
function restart-apiserver {
  ssh-to-node "$1" "sudo systemctl restart kube-apiserver"
}

# Perform preparations required to run e2e tests
function prepare-e2e() {
    echo "libvirt-coreos doesn't need special preparations for e2e tests" 1>&2
}
