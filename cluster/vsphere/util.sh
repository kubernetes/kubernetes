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

# A library of helper functions and constants for the local config.

# Use the config file specified in $KUBE_CONFIG_FILE, or default to
# config-default.sh.

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
source "${KUBE_ROOT}/cluster/vsphere/config-common.sh"
source "${KUBE_ROOT}/cluster/vsphere/${KUBE_CONFIG_FILE-"config-default.sh"}"
source "${KUBE_ROOT}/cluster/common.sh"

# Detect the IP for the master
#
# Assumed vars:
#   MASTER_NAME
# Vars set:
#   KUBE_MASTER
#   KUBE_MASTER_IP

function detect-master {
  KUBE_MASTER=${MASTER_NAME}
  if [[ -z "${KUBE_MASTER_IP-}" ]]; then
    KUBE_MASTER_IP=$(govc vm.ip ${MASTER_NAME})
  fi
  if [[ -z "${KUBE_MASTER_IP-}" ]]; then
    echo "Could not detect Kubernetes master node. Make sure you've launched a cluster with 'kube-up.sh'" >&2
    exit 1
  fi
  echo "Using master: $KUBE_MASTER (external IP: $KUBE_MASTER_IP)"
}

# Detect the information about the nodes
#
# Assumed vars:
#   NODE_NAMES
# Vars set:
#   KUBE_NODE_IP_ADDRESS (array)
function detect-nodes {
  KUBE_NODE_IP_ADDRESSES=()
  for (( i=0; i<${#NODE_NAMES[@]}; i++)); do
    local nodeip=$(govc vm.ip ${NODE_NAMES[$i]})
    if [[ -z "${nodeip-}" ]] ; then
      echo "Did not find ${NODE_NAMES[$i]}" >&2
    else
      echo "Found ${NODE_NAMES[$i]} at ${nodeip}"
      KUBE_NODE_IP_ADDRESSES+=("${nodeip}")
    fi
  done
  if [[ -z "${KUBE_NODE_IP_ADDRESSES-}" ]]; then
    echo "Could not detect Kubernetes nodes. Make sure you've launched a cluster with 'kube-up.sh'" >&2
    exit 1
  fi
}

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

function verify-prereqs {
  which "govc" >/dev/null || {
    echo "Can't find govc in PATH, please install and retry."
    echo ""
    echo "    go install github.com/vmware/govmomi/govc"
    echo ""
    exit 1
  }
}

function verify-ssh-prereqs {
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

  # Expect at least one identity to be available.
  if ! ssh-add -L 1> /dev/null 2> /dev/null; then
    echo "Could not find or add an SSH identity."
    echo "Please start ssh-agent, add your identity, and retry."
    exit 1
  fi
}

# Create a temp dir that'll be deleted at the end of this bash session.
#
# Vars set:
#   KUBE_TEMP
function ensure-temp-dir {
  if [[ -z ${KUBE_TEMP-} ]]; then
    KUBE_TEMP=$(mktemp -d -t kubernetes.XXXXXX)
    trap-add 'rm -rf "${KUBE_TEMP}"' EXIT
  fi
}

# Take the local tar files and upload them to the master.
#
# Assumed vars:
#   MASTER_NAME
#   SERVER_BINARY_TAR
#   SALT_TAR
function upload-server-tars {
  local vm_ip

  vm_ip=$(govc vm.ip "${MASTER_NAME}")
  kube-ssh ${vm_ip} "mkdir -p /home/kube/cache/kubernetes-install"

  local tar
  for tar in "${SERVER_BINARY_TAR}" "${SALT_TAR}"; do
    kube-scp ${vm_ip} "${tar}" "/home/kube/cache/kubernetes-install/${tar##*/}"
  done
}

# Run command over ssh
function kube-ssh {
  local host="$1"
  shift
  ssh ${SSH_OPTS-} "kube@${host}" "$@" 2> /dev/null
}

# Copy file over ssh
function kube-scp {
  local host="$1"
  local src="$2"
  local dst="$3"
  scp ${SSH_OPTS-} "${src}" "kube@${host}:${dst}"
}

# Instantiate a generic kubernetes virtual machine (master or node)
#
# Usage:
#   kube-up-vm VM_NAME [options to pass to govc vm.create]
#
# Example:
#   kube-up-vm "vm-name" -c 2 -m 4096
#
# Assumed vars:
#   DISK
#   GUEST_ID
function kube-up-vm {
  local vm_name="$1"
  shift

  govc vm.create \
    -debug \
    -disk="${DISK}" \
    -g="${GUEST_ID}" \
    -link=true \
    "$@" \
    "${vm_name}"

  # Retrieve IP first, to confirm the guest operations agent is running.
  govc vm.ip "${vm_name}" > /dev/null

  govc guest.mkdir \
    -l "kube:kube" \
    -vm="${vm_name}" \
    -p \
    /home/kube/.ssh

  ssh-add -L > "${KUBE_TEMP}/${vm_name}-authorized_keys"

  govc guest.upload \
    -l "kube:kube" \
    -vm="${vm_name}" \
    -f \
    "${KUBE_TEMP}/${vm_name}-authorized_keys" \
    /home/kube/.ssh/authorized_keys
}

# Kick off a local script on a kubernetes virtual machine (master or node)
#
# Usage:
#   kube-run VM_NAME LOCAL_FILE
function kube-run {
  local vm_name="$1"
  local file="$2"
  local dst="/tmp/$(basename "${file}")"
  govc guest.upload -l "kube:kube" -vm="${vm_name}" -f -perm=0755 "${file}" "${dst}"
  echo "uploaded ${file} to ${dst}"
  local vm_ip
  vm_ip=$(govc vm.ip "${vm_name}")
  kube-ssh ${vm_ip} "nohup sudo ${dst} < /dev/null 1> ${dst}.out 2> ${dst}.err &"
}

#
# run the command remotely and check if the specific kube artifact is running or not.
# keep checking till the you hit the timeout. default timeout 300s
#
# Usage:
#   kube_check 10.0.0.1 cmd timeout
function kube-check {
  nodeip=$1
  cmd=$2
  sleepstep=5
  if [[ $# -lt 3 || -z $3 ]]; then
    timeout=300
  else
    timeout=$3
  fi
  let effective_timeout=($timeout/$sleepstep)
  attempt=0
  echo
  printf "This may take several minutes. Bound to $effective_timeout attempts"
  while true; do
    local rc=0
    output=$(kube-ssh ${nodeip} "${cmd}") || rc=1
    if [[ $rc != 0 ]]; then
      if (( $attempt == $effective_timeout )); then
        echo
        echo "(Failed) rc: $rc Output: ${output}"
        echo
        echo -e "${cmd} failed to start on ${nodeip}. Your cluster is unlikely" >&2
        echo "to work correctly. You may have to debug it by logging in." >&2
        echo
        exit 1
      fi
    else
      echo
      echo -e "[${cmd}] passed"
      echo
      break
    fi
    printf "."
    attempt=$(($attempt+1))
    sleep $sleepstep
  done
}

#
# verify if salt master is up. check 30 times and then echo out bad output and return 0
#
# Usage:
#   remote-pgrep 10.0.0.1 salt-master
#
function remote-pgrep {
  nodeip=$1
  regex=$2

  max_attempt=60

  printf "This may take several minutes. Bound to $max_attempt attempts"

  attempt=0
  while true; do
    local rc=0
    output=$(kube-ssh ${nodeip} pgrep ${regex}) || rc=1

    if [[ $rc != 0 ]]; then
      if (( $attempt == $max_attempt )); then
        echo
        echo "(Failed) rc: $rc, output:${output}"
        echo
        echo -e "${regex} failed to start on ${nodeip} after checking for $attempt attempts. Your cluster is unlikely" >&2
        echo "to work correctly. You may have to debug it by logging in." >&2
        echo
        exit 1
      fi
    else
      echo
      echo -e "[${regex} running]"
      echo
      break
    fi
    printf "."
    attempt=$(($attempt+1))
    sleep 10
  done
}

# identify the pod routes and route them together.
#
# Assumptions:
#  All packages have been installed and kubelet has started running.
#
function setup-pod-routes {
  # wait till the kubelet sets up the bridge.
  echo "Setting up routes"
  for (( i=0; i<${#NODE_NAMES[@]}; i++)); do
     printf "check if cbr0 bridge is ready on ${NODE_NAMES[$i]}\n"
     kube-check ${KUBE_NODE_IP_ADDRESSES[$i]} 'sudo ifconfig cbr0 | grep -oP "inet addr:\K\S+"'
  done


  # identify the subnet assigned to the node by the kubernertes controller manager.
  KUBE_NODE_BRIDGE_NETWORK=()
  for (( i=0; i<${#NODE_NAMES[@]}; i++)); do
     printf " finding network of cbr0 bridge on node  ${NODE_NAMES[$i]}\n"
     network=$(kube-ssh ${KUBE_NODE_IP_ADDRESSES[$i]} 'sudo ip route show | grep -E "dev cbr0" | cut -d     " " -f1')
     KUBE_NODE_BRIDGE_NETWORK+=("${network}")
  done


  # Make the pods visible to each other and to the master.
  # The master needs have routes to the pods for the UI to work.
  local j
  for (( i=0; i<${#NODE_NAMES[@]}; i++)); do
     printf "setting up routes for ${NODE_NAMES[$i]}"
     kube-ssh "${KUBE_MASTER_IP}" "sudo route add -net ${KUBE_NODE_BRIDGE_NETWORK[${i}]} gw ${KUBE_NODE_IP_ADDRESSES[${i}]}"
     for (( j=0; j<${#NODE_NAMES[@]}; j++)); do
        if [[ $i != $j ]]; then
           kube-ssh ${KUBE_NODE_IP_ADDRESSES[$i]} "sudo route add -net ${KUBE_NODE_BRIDGE_NETWORK[$j]} gw ${KUBE_NODE_IP_ADDRESSES[$j]}"
        fi
      done
  done
}

# Instantiate a kubernetes cluster
#
# Assumed vars:
#   KUBE_ROOT
#   <Various vars set in config file>
function kube-up {
  verify-ssh-prereqs
  find-release-tars

  ensure-temp-dir

  load-or-gen-kube-basicauth
  python "${KUBE_ROOT}/third_party/htpasswd/htpasswd.py" \
    -b -c "${KUBE_TEMP}/htpasswd" "$KUBE_USER" "$KUBE_PASSWORD"
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

  echo "Starting master VM (this can take a minute)..."

  (
    echo "#! /bin/bash"
    echo "readonly MY_NAME=${MASTER_NAME}"
    grep -v "^#" "${KUBE_ROOT}/cluster/vsphere/templates/hostname.sh"
    echo "cd /home/kube/cache/kubernetes-install"
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
    grep -v "^#" "${KUBE_ROOT}/cluster/vsphere/templates/create-dynamic-salt-files.sh"
    grep -v "^#" "${KUBE_ROOT}/cluster/vsphere/templates/install-release.sh"
    grep -v "^#" "${KUBE_ROOT}/cluster/vsphere/templates/salt-master.sh"
  ) > "${KUBE_TEMP}/master-start.sh"

  kube-up-vm ${MASTER_NAME} -c ${MASTER_CPU-1} -m ${MASTER_MEMORY_MB-1024}
  upload-server-tars
  kube-run ${MASTER_NAME} "${KUBE_TEMP}/master-start.sh"
  # Print master IP, so user can log in for debugging.
  detect-master
  echo

  echo "Starting node VMs (this can take a minute)..."
  for (( i=0; i<${#NODE_NAMES[@]}; i++)); do
    (
      echo "#! /bin/bash"
      echo "readonly MY_NAME=${NODE_NAMES[$i]}"
      grep -v "^#" "${KUBE_ROOT}/cluster/vsphere/templates/hostname.sh"
      echo "KUBE_MASTER=${KUBE_MASTER}"
      echo "KUBE_MASTER_IP=${KUBE_MASTER_IP}"
      echo "NODE_IP_RANGE=$NODE_IP_RANGES"
      grep -v "^#" "${KUBE_ROOT}/cluster/vsphere/templates/salt-minion.sh"
    ) > "${KUBE_TEMP}/node-start-${i}.sh"

    (
      kube-up-vm "${NODE_NAMES[$i]}" -c ${NODE_CPU-1} -m ${NODE_MEMORY_MB-1024}
      kube-run "${NODE_NAMES[$i]}" "${KUBE_TEMP}/node-start-${i}.sh"
    ) &
  done

  local fail=0
  local job
  for job in $(jobs -p); do
      wait "${job}" || fail=$((fail + 1))
  done
  if (( $fail != 0 )); then
    echo "${fail} commands failed.  Exiting." >&2
    exit 2
  fi

  # Print node IPs, so user can log in for debugging.
  detect-nodes

  printf "Waiting for salt-master to be up on ${KUBE_MASTER} ...\n"
  remote-pgrep ${KUBE_MASTER_IP} "salt-master"

  printf "Waiting for all packages to be installed on ${KUBE_MASTER} ...\n"
  kube-check  ${KUBE_MASTER_IP} 'sudo salt "kubernetes-master" state.highstate -t 30 | grep -E "Failed:[[:space:]]+0"'

  local i
  for (( i=0; i<${#NODE_NAMES[@]}; i++)); do
    printf "Waiting for salt-minion to be up on ${NODE_NAMES[$i]} ....\n"
    remote-pgrep ${KUBE_NODE_IP_ADDRESSES[$i]} "salt-minion"
    printf "Waiting for all salt packages to be installed on ${NODE_NAMES[$i]} .... \n"
    kube-check  ${KUBE_MASTER_IP} 'sudo salt '"${NODE_NAMES[$i]}"' state.highstate -t 30 | grep -E "Failed:[[:space:]]+0"'
    printf " OK\n"
  done

  echo

  echo "Waiting for master and node initialization."
  echo
  echo "  This will continually check to see if the API for kubernetes is reachable."
  echo "  This might loop forever if there was some uncaught error during start up."
  echo

  until curl --insecure --user "${KUBE_USER}:${KUBE_PASSWORD}" --max-time 5 \
          --fail --output /dev/null --silent "https://${KUBE_MASTER_IP}/healthz"; do
      printf "."
      sleep 2
  done
  printf " OK\n"

  for (( i=0; i<${#NODE_NAMES[@]}; i++)); do
    printf "Waiting for ${NODE_NAMES[$i]} to become available..."
    until curl --max-time 5 \
            --fail --output /dev/null --silent "http://${KUBE_NODE_IP_ADDRESSES[$i]}:10250/healthz"; do
        printf "."
        sleep 2
    done
    printf " OK\n"
  done

  setup-pod-routes

  echo "Kubernetes cluster created."
  # TODO use token instead of basic auth
  export KUBE_CERT="/tmp/$RANDOM-kubecfg.crt"
  export KUBE_KEY="/tmp/$RANDOM-kubecfg.key"
  export CA_CERT="/tmp/$RANDOM-kubernetes.ca.crt"
  export CONTEXT="vsphere_${INSTANCE_PREFIX}"

  (
    umask 077

    kube-ssh "${KUBE_MASTER_IP}" sudo cat /srv/kubernetes/kubecfg.crt >"${KUBE_CERT}" 2>/dev/null
    kube-ssh "${KUBE_MASTER_IP}" sudo cat /srv/kubernetes/kubecfg.key >"${KUBE_KEY}" 2>/dev/null
    kube-ssh "${KUBE_MASTER_IP}" sudo cat /srv/kubernetes/ca.crt >"${CA_CERT}" 2>/dev/null

    create-kubeconfig
  )
  printf "\n"

  echo
  echo "Sanity checking cluster..."

  sleep 5

  # Basic sanity checking
  for (( i=0; i<${#NODE_NAMES[@]}; i++)); do
      # Make sure docker is installed
      kube-ssh "${KUBE_NODE_IP_ADDRESSES[$i]}" which docker > /dev/null || {
        echo "Docker failed to install on ${NODE_NAMES[$i]}. Your cluster is unlikely" >&2
        echo "to work correctly. Please run ./cluster/kube-down.sh and re-create the" >&2
        echo "cluster. (sorry!)" >&2
        exit 1
      }
  done

  # ensures KUBECONFIG is set
  get-kubeconfig-basicauth
  echo
  echo "Kubernetes cluster is running. The master is running at:"
  echo
  echo "  https://${KUBE_MASTER_IP}"
  echo
  echo "The user name and password to use is located in ${KUBECONFIG}"
  echo
}

# Delete a kubernetes cluster
function kube-down {
  govc vm.destroy ${MASTER_NAME} &

  for (( i=0; i<${#NODE_NAMES[@]}; i++)); do
    govc vm.destroy ${NODE_NAMES[i]} &
  done

  wait
}

# Update a kubernetes cluster with latest source
function kube-push {
  verify-ssh-prereqs
  find-release-tars

  detect-master
  upload-server-tars

  (
    echo "#! /bin/bash"
    echo "cd /home/kube/cache/kubernetes-install"
    echo "readonly SERVER_BINARY_TAR='${SERVER_BINARY_TAR##*/}'"
    echo "readonly SALT_TAR='${SALT_TAR##*/}'"
    grep -v "^#" "${KUBE_ROOT}/cluster/vsphere/templates/install-release.sh"
    echo "echo Executing configuration"
    echo "sudo salt '*' mine.update"
    echo "sudo salt --force-color '*' state.highstate"
  ) | kube-ssh "${KUBE_MASTER_IP}"

  get-kubeconfig-basicauth

  echo
  echo "Kubernetes cluster is running.  The master is running at:"
  echo
  echo "  https://${KUBE_MASTER_IP}"
  echo
  echo "The user name and password to use is located in ${KUBECONFIG:-$DEFAULT_KUBECONFIG}."
  echo
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
	echo "TODO"
}
