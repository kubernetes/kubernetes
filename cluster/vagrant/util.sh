#!/bin/bash

# Copyright 2014 The Kubernetes Authors.
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
source "${KUBE_ROOT}/cluster/common.sh"

function detect-master () {
  KUBE_MASTER_IP=$MASTER_IP
  echo "KUBE_MASTER_IP: ${KUBE_MASTER_IP}" 1>&2
}

# Get node IP addresses and store in KUBE_NODE_IP_ADDRESSES[]
function detect-nodes {
  echo "Nodes already detected" 1>&2
  KUBE_NODE_IP_ADDRESSES=("${NODE_IPS[@]}")
}

# Verify prereqs on host machine  Also sets exports USING_KUBE_SCRIPTS=true so
# that our Vagrantfile doesn't error out.
function verify-prereqs {
  for x in vagrant; do
    if ! which "$x" >/dev/null; then
      echo "Can't find $x in PATH, please fix and retry."
      exit 1
    fi
  done

  local vagrant_plugins=$(vagrant plugin list | sed '-es% .*$%%' '-es%  *% %g' | tr ' ' $'\n')
  local providers=(
      # Format is:
      #   provider_ctl_executable vagrant_provider_name vagrant_provider_plugin_re
      # either provider_ctl_executable or vagrant_provider_plugin_re can
      # be blank (i.e., '') if none is needed by Vagrant (see, e.g.,
      # virtualbox entry)
      '' vmware_fusion vagrant-vmware-fusion
      '' vmware_workstation vagrant-vmware-workstation
      prlctl parallels vagrant-parallels
      VBoxManage virtualbox ''
      virsh libvirt vagrant-libvirt
      '' vsphere vagrant-vsphere
  )
  local provider_found=''
  local provider_bin
  local provider_name
  local provider_plugin_re

  while [ "${#providers[@]}" -gt 0 ]; do
    provider_bin=${providers[0]}
    provider_name=${providers[1]}
    provider_plugin_re=${providers[2]}
    providers=("${providers[@]:3}")

    # If the provider is explicitly set, look only for that provider
    if [ -n "${VAGRANT_DEFAULT_PROVIDER:-}" ] \
        && [ "${VAGRANT_DEFAULT_PROVIDER}" != "${provider_name}" ]; then
      continue
    fi

    if ([ -z "${provider_bin}" ] \
          || which "${provider_bin}" >/dev/null 2>&1) \
        && ([ -z "${provider_plugin_re}" ] \
          || [ -n "$(echo "${vagrant_plugins}" | grep -E "^${provider_plugin_re}$")" ]); then
      provider_found="${provider_name}"
      # Stop after finding the first viable provider
      break
    fi
  done

  if [ -z "${provider_found}" ]; then
    if [ -n "${VAGRANT_DEFAULT_PROVIDER:-}" ]; then
      echo "Can't find the necessary components for the ${VAGRANT_DEFAULT_PROVIDER} vagrant provider."
      echo "Possible reasons could be: "
      echo -e "\t- vmrun utility is not in your path"
      echo -e "\t- Vagrant plugin was not found."
      echo -e "\t- VAGRANT_DEFAULT_PROVIDER is set, but not found."
      echo "Please fix and retry."
    else
      echo "Can't find the necessary components for any viable vagrant providers (e.g., virtualbox), please fix and retry."
    fi

    exit 1
  fi

  # Set VAGRANT_CWD to KUBE_ROOT so that we find the right Vagrantfile no
  # matter what directory the tools are called from.
  export VAGRANT_CWD="${KUBE_ROOT}"

  export USING_KUBE_SCRIPTS=true
}

# Create a set of provision scripts for the master and each of the nodes
function create-provision-scripts {
  kube::util::ensure-temp-dir

  (
    echo "#! /bin/bash"
    echo-kube-env
    echo "NODE_IP='${MASTER_IP}'"
    echo "CONTAINER_ADDR='${MASTER_CONTAINER_ADDR}'"
    echo "CONTAINER_NETMASK='${MASTER_CONTAINER_NETMASK}'"
    awk '!/^#/' "${KUBE_ROOT}/cluster/vagrant/provision-utils.sh"
    awk '!/^#/' "${KUBE_ROOT}/cluster/vagrant/provision-network-master.sh"
    awk '!/^#/' "${KUBE_ROOT}/cluster/vagrant/provision-master.sh"
  ) > "${KUBE_TEMP}/master-start.sh"

  for (( i=0; i<${#NODE_NAMES[@]}; i++)); do
    (
      echo "#! /bin/bash"
      echo-kube-env
      echo "NODE_NAME=(${NODE_NAMES[$i]})"
      echo "NODE_IP='${NODE_IPS[$i]}'"
      echo "NODE_ID='$i'"
      echo "CONTAINER_ADDR='${NODE_CONTAINER_ADDRS[$i]}'"
      echo "CONTAINER_NETMASK='${NODE_CONTAINER_NETMASKS[$i]}'"
      awk '!/^#/' "${KUBE_ROOT}/cluster/vagrant/provision-utils.sh"
      awk '!/^#/' "${KUBE_ROOT}/cluster/vagrant/provision-network-node.sh"
      awk '!/^#/' "${KUBE_ROOT}/cluster/vagrant/provision-node.sh"
    ) > "${KUBE_TEMP}/node-start-${i}.sh"
  done
}

function echo-kube-env() {
  echo "KUBE_ROOT=/vagrant"
  echo "INSTANCE_PREFIX='${INSTANCE_PREFIX}'"
  echo "MASTER_NAME='${INSTANCE_PREFIX}-master'"
  echo "MASTER_IP='${MASTER_IP}'"
  echo "NODE_NAMES=(${NODE_NAMES[@]})"
  echo "NODE_IPS=(${NODE_IPS[@]})"
  echo "DEFAULT_NETWORK_IF_NAME=${DEFAULT_NETWORK_IF_NAME}"
  echo "CONTAINER_SUBNET='${CONTAINER_SUBNET}'"
  echo "CLUSTER_IP_RANGE='${CLUSTER_IP_RANGE}'"
  echo "MASTER_CONTAINER_SUBNET='${MASTER_CONTAINER_SUBNET}'"
  echo "NODE_CONTAINER_NETMASKS='${NODE_CONTAINER_NETMASKS[@]}'"
  echo "NODE_CONTAINER_SUBNETS=(${NODE_CONTAINER_SUBNETS[@]})"
  echo "SERVICE_CLUSTER_IP_RANGE='${SERVICE_CLUSTER_IP_RANGE}'"
  echo "MASTER_USER='${MASTER_USER}'"
  echo "MASTER_PASSWD='${MASTER_PASSWD}'"
  echo "KUBE_USER='${KUBE_USER}'"
  echo "KUBE_PASSWORD='${KUBE_PASSWORD}'"
  echo "KUBE_BEARER_TOKEN='${KUBE_BEARER_TOKEN}'"
  echo "ENABLE_CLUSTER_MONITORING='${ENABLE_CLUSTER_MONITORING}'"
  echo "ENABLE_CLUSTER_LOGGING='${ENABLE_CLUSTER_LOGGING:-false}'"
  echo "ELASTICSEARCH_LOGGING_REPLICAS='${ELASTICSEARCH_LOGGING_REPLICAS:-1}'"
  echo "ENABLE_NODE_LOGGING='${ENABLE_NODE_LOGGING:-false}'"
  echo "ENABLE_CLUSTER_UI='${ENABLE_CLUSTER_UI}'"
  echo "ENABLE_HOSTPATH_PROVISIONER='${ENABLE_HOSTPATH_PROVISIONER:-false}'"
  echo "LOGGING_DESTINATION='${LOGGING_DESTINATION:-}'"
  echo "ENABLE_CLUSTER_DNS='${ENABLE_CLUSTER_DNS:-false}'"
  echo "DNS_SERVER_IP='${DNS_SERVER_IP:-}'"
  echo "DNS_DOMAIN='${DNS_DOMAIN:-}'"
  echo "RUNTIME_CONFIG='${RUNTIME_CONFIG:-}'"
  echo "ADMISSION_CONTROL='${ADMISSION_CONTROL:-}'"
  echo "DOCKER_OPTS='${EXTRA_DOCKER_OPTS:-}'"
  echo "VAGRANT_DEFAULT_PROVIDER='${VAGRANT_DEFAULT_PROVIDER:-}'"
  echo "KUBELET_TOKEN='${KUBELET_TOKEN:-}'"
  echo "KUBE_PROXY_TOKEN='${KUBE_PROXY_TOKEN:-}'"
  echo "MASTER_EXTRA_SANS='${MASTER_EXTRA_SANS:-}'"
  echo "ENABLE_CPU_CFS_QUOTA='${ENABLE_CPU_CFS_QUOTA}'"
  echo "NETWORK_PROVIDER='${NETWORK_PROVIDER:-}'"
  echo "OPENCONTRAIL_TAG='${OPENCONTRAIL_TAG:-}'"
  echo "OPENCONTRAIL_KUBERNETES_TAG='${OPENCONTRAIL_KUBERNETES_TAG:-}'"
  echo "OPENCONTRAIL_PUBLIC_SUBNET='${OPENCONTRAIL_PUBLIC_SUBNET:-}'"
  echo "E2E_STORAGE_TEST_ENVIRONMENT='${E2E_STORAGE_TEST_ENVIRONMENT:-}'"
  echo "CUSTOM_FEDORA_REPOSITORY_URL='${CUSTOM_FEDORA_REPOSITORY_URL:-}'"
  echo "EVICTION_HARD='${EVICTION_HARD:-}'"
}

function verify-cluster {
  # TODO: How does the user know the difference between "tak[ing] some
  # time" and "loop[ing] forever"? Can we give more specific feedback on
  # whether "an error" has occurred?
  echo "Each machine instance has been created/updated."
  echo "  Now waiting for the Salt provisioning process to complete on each machine."
  echo "  This can take some time based on your network, disk, and cpu speed."
  echo "  It is possible for an error to occur during Salt provision of cluster and this could loop forever."

  # verify master has all required daemons
  echo "Validating master"
  local machine="master"
  local -a required_processes=("kube-apiserver" "kube-scheduler" "kube-controller-manager" "kubelet" "docker")
  local validated="1"
  until [[ "$validated" == "0" ]]; do
    validated="0"
    for process in "${required_processes[@]}"; do
      vagrant ssh "${machine}" -c "pgrep -f ${process}" >/dev/null 2>&1 || {
        printf "."
        validated="1"
        sleep 2
      }
    done
  done

  # verify each node has all required daemons
  local i
  for (( i=0; i<${#NODE_NAMES[@]}; i++)); do
    echo "Validating ${VAGRANT_NODE_NAMES[$i]}"
    local machine=${VAGRANT_NODE_NAMES[$i]}
    local -a required_processes=("kube-proxy" "kubelet" "docker")
    local validated="1"
    until [[ "${validated}" == "0" ]]; do
      validated="0"
      for process in "${required_processes[@]}"; do
        vagrant ssh "${machine}" -c "pgrep -f ${process}" >/dev/null 2>&1 || {
          printf "."
          validated="1"
          sleep 2
        }
      done
    done
  done

  echo
  echo "Waiting for each node to be registered with cloud provider"
  for (( i=0; i<${#NODE_NAMES[@]}; i++)); do
    local validated="0"
    start="$(date +%s)"
    until [[ "$validated" == "1" ]]; do
      now="$(date +%s)"
      # Timeout set to 3 minutes
      if [ $((now - start)) -gt 180 ]; then
        echo "Timeout while waiting for echo node to be registered with cloud provider"
        exit 2
      fi
      local nodes=$("${KUBE_ROOT}/cluster/kubectl.sh" get nodes -o name)
      validated=$(echo $nodes | grep -c "${NODE_NAMES[i]}") || {
        printf "."
        sleep 2
        validated="0"
      }
    done
  done

  # By this time, all kube api calls should work, so no need to loop and retry.
  echo "Validating we can run kubectl commands."
  vagrant ssh master --command "kubectl get pods" || {
    echo "WARNING: kubectl to localhost failed.  This could mean localhost is not bound to an IP"
  }

  (
    # ensures KUBECONFIG is set
    get-kubeconfig-basicauth
    get-kubeconfig-bearertoken
    echo
    echo "Kubernetes cluster is running."
    echo
    echo "The master is running at:"
    echo
    echo "  https://${MASTER_IP}"
    echo
    echo "Administer and visualize its resources using Cockpit:"
    echo
    echo "  https://${MASTER_IP}:9090"
    echo
    echo "For more information on Cockpit, visit http://cockpit-project.org"
    echo
    echo "The user name and password to use is located in ${KUBECONFIG}"
    echo
  )
}

# Instantiate a kubernetes cluster
function kube-up {
  load-or-gen-kube-basicauth
  load-or-gen-kube-bearertoken
  get-tokens
  create-provision-scripts

  vagrant up --no-parallel

  export KUBE_CERT="/tmp/$RANDOM-kubecfg.crt"
  export KUBE_KEY="/tmp/$RANDOM-kubecfg.key"
  export CA_CERT="/tmp/$RANDOM-kubernetes.ca.crt"
  export CONTEXT="vagrant"

  (
   umask 077
   vagrant ssh master -- sudo cat /srv/kubernetes/kubecfg.crt >"${KUBE_CERT}" 2>/dev/null
   vagrant ssh master -- sudo cat /srv/kubernetes/kubecfg.key >"${KUBE_KEY}" 2>/dev/null
   vagrant ssh master -- sudo cat /srv/kubernetes/ca.crt >"${CA_CERT}" 2>/dev/null

   # Update the user's kubeconfig to include credentials for this apiserver.
   create-kubeconfig
  )

  verify-cluster
}

# Delete a kubernetes cluster
function kube-down {
  vagrant destroy -f
}

# Update a kubernetes cluster with latest source
function kube-push {
  get-kubeconfig-basicauth
  get-kubeconfig-bearertoken
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
  "${KUBE_ROOT}/cluster/kube-up.sh"
  echo "Vagrant test setup complete" 1>&2
}

# Execute after running tests to perform any required clean-up
function test-teardown {
  kube-down
}

# Find the node name based on the IP address
function find-vagrant-name-by-ip {
  local ip="$1"
  local ip_pattern="${NODE_IP_BASE}(.*)"

  # This is subtle.  We map 10.245.2.2 -> node-1.  We do this by matching a
  # regexp and using the capture to construct the name.
  [[ $ip =~ $ip_pattern ]] || {
    return 1
  }

  echo "node-$((${BASH_REMATCH[1]} - 1))"
}

# Find the vagrant machine name based on the host name of the node
function find-vagrant-name-by-node-name {
  local ip="$1"
  if [[ "$ip" == "${INSTANCE_PREFIX}-master" ]]; then
    echo "master"
    return $?
  fi
  local ip_pattern="${INSTANCE_PREFIX}-node-(.*)"

  [[ $ip =~ $ip_pattern ]] || {
    return 1
  }

  echo "node-${BASH_REMATCH[1]}"
}


# SSH to a node by name or IP ($1) and run a command ($2).
function ssh-to-node {
  local node="$1"
  local cmd="$2"
  local machine

  machine=$(find-vagrant-name-by-ip $node) || true
  [[ -n ${machine-} ]] || machine=$(find-vagrant-name-by-node-name $node) || true
  [[ -n ${machine-} ]] || {
    echo "Cannot find machine to ssh to: $1"
    return 1
  }

  vagrant ssh "${machine}" -c "${cmd}"
}

# Perform preparations required to run e2e tests
function prepare-e2e() {
  echo "Vagrant doesn't need special preparations for e2e tests" 1>&2
}

function get-tokens() {
  KUBELET_TOKEN=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
  KUBE_PROXY_TOKEN=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
}
