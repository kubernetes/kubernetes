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

# A library of helper functions and constants for the local config.

# Use the config file specified in $LMKTFY_CONFIG_FILE, or default to
# config-default.sh.
LMKTFY_ROOT=$(dirname "${BASH_SOURCE}")/../..
source "${LMKTFY_ROOT}/cluster/vsphere/config-common.sh"
source "${LMKTFY_ROOT}/cluster/vsphere/${LMKTFY_CONFIG_FILE-"config-default.sh"}"

# Detect the IP for the master
#
# Assumed vars:
#   MASTER_NAME
# Vars set:
#   LMKTFY_MASTER
#   LMKTFY_MASTER_IP
function detect-master {
  LMKTFY_MASTER=${MASTER_NAME}
  if [[ -z "${LMKTFY_MASTER_IP-}" ]]; then
    LMKTFY_MASTER_IP=$(govc vm.ip ${MASTER_NAME})
  fi
  if [[ -z "${LMKTFY_MASTER_IP-}" ]]; then
    echo "Could not detect LMKTFY master node. Make sure you've launched a cluster with 'lmktfy-up.sh'" >&2
    exit 1
  fi
  echo "Using master: $LMKTFY_MASTER (external IP: $LMKTFY_MASTER_IP)"
}

# Detect the information about the minions
#
# Assumed vars:
#   MINION_NAMES
# Vars set:
#   LMKTFY_MINION_IP_ADDRESS (array)
function detect-minions {
  LMKTFY_MINION_IP_ADDRESSES=()
  for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
    local minion_ip=$(govc vm.ip ${MINION_NAMES[$i]})
    if [[ -z "${minion_ip-}" ]] ; then
      echo "Did not find ${MINION_NAMES[$i]}" >&2
    else
      echo "Found ${MINION_NAMES[$i]} at ${minion_ip}"
      LMKTFY_MINION_IP_ADDRESSES+=("${minion_ip}")
    fi
  done
  if [[ -z "${LMKTFY_MINION_IP_ADDRESSES-}" ]]; then
    echo "Could not detect LMKTFY minion nodes. Make sure you've launched a cluster with 'lmktfy-up.sh'" >&2
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
#   LMKTFY_TEMP
function ensure-temp-dir {
  if [[ -z ${LMKTFY_TEMP-} ]]; then
    LMKTFY_TEMP=$(mktemp -d -t lmktfy.XXXXXX)
    trap-add 'rm -rf "${LMKTFY_TEMP}"' EXIT
  fi
}

# Verify and find the various tar files that we are going to use on the server.
#
# Vars set:
#   SERVER_BINARY_TAR
#   SALT_TAR
function find-release-tars {
  SERVER_BINARY_TAR="${LMKTFY_ROOT}/server/lmktfy-server-linux-amd64.tar.gz"
  if [[ ! -f "$SERVER_BINARY_TAR" ]]; then
    SERVER_BINARY_TAR="${LMKTFY_ROOT}/_output/release-tars/lmktfy-server-linux-amd64.tar.gz"
  fi
  if [[ ! -f "$SERVER_BINARY_TAR" ]]; then
    echo "!!! Cannot find lmktfy-server-linux-amd64.tar.gz"
    exit 1
  fi

  SALT_TAR="${LMKTFY_ROOT}/server/lmktfy-salt.tar.gz"
  if [[ ! -f "$SALT_TAR" ]]; then
    SALT_TAR="${LMKTFY_ROOT}/_output/release-tars/lmktfy-salt.tar.gz"
  fi
  if [[ ! -f "$SALT_TAR" ]]; then
    echo "!!! Cannot find lmktfy-salt.tar.gz"
    exit 1
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
  lmktfy-ssh ${vm_ip} "mkdir -p /home/lmktfy/cache/lmktfy-install"

  local tar
  for tar in "${SERVER_BINARY_TAR}" "${SALT_TAR}"; do
    lmktfy-scp ${vm_ip} "${tar}" "/home/lmktfy/cache/lmktfy-install/${tar##*/}"
  done
}

# Ensure that we have a password created for validating to the master. Will
# read from $HOME/.lmktfy_auth if available.
#
# Vars set:
#   LMKTFY_USER
#   LMKTFY_PASSWORD
function get-password {
  local file="$HOME/.lmktfy_auth"
  if [[ -r "$file" ]]; then
    LMKTFY_USER=$(cat "$file" | python -c 'import json,sys;print json.load(sys.stdin)["User"]')
    LMKTFY_PASSWORD=$(cat "$file" | python -c 'import json,sys;print json.load(sys.stdin)["Password"]')
    return
  fi
  LMKTFY_USER=admin
  LMKTFY_PASSWORD=$(python -c 'import string,random; print "".join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(16))')

  # Store password for reuse.
  cat << EOF > "$file"
{
  "User": "$LMKTFY_USER",
  "Password": "$LMKTFY_PASSWORD"
}
EOF
  chmod 0600 "$file"
}

# Run command over ssh
function lmktfy-ssh {
  local host="$1"
  shift
  ssh ${SSH_OPTS-} "lmktfy@${host}" "$@" 2> /dev/null
}

# Copy file over ssh
function lmktfy-scp {
  local host="$1"
  local src="$2"
  local dst="$3"
  scp ${SSH_OPTS-} "${src}" "lmktfy@${host}:${dst}"
}

# Instantiate a generic lmktfy virtual machine (master or minion)
#
# Usage:
#   lmktfy-up-vm VM_NAME [options to pass to govc vm.create]
#
# Example:
#   lmktfy-up-vm "vm-name" -c 2 -m 4096
#
# Assumed vars:
#   DISK
#   GUEST_ID
function lmktfy-up-vm {
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
    -vm="${vm_name}" \
    -p \
    /home/lmktfy/.ssh

  ssh-add -L > "${LMKTFY_TEMP}/${vm_name}-authorized_keys"

  govc guest.upload \
    -vm="${vm_name}" \
    -f \
    "${LMKTFY_TEMP}/${vm_name}-authorized_keys" \
    /home/lmktfy/.ssh/authorized_keys
}

# Kick off a local script on a lmktfy virtual machine (master or minion)
#
# Usage:
#   lmktfy-run VM_NAME LOCAL_FILE
function lmktfy-run {
  local vm_name="$1"
  local file="$2"
  local dst="/tmp/$(basename "${file}")"
  govc guest.upload -vm="${vm_name}" -f -perm=0755 "${file}" "${dst}"

  local vm_ip
  vm_ip=$(govc vm.ip "${vm_name}")
  lmktfy-ssh ${vm_ip} "nohup sudo ${dst} < /dev/null 1> ${dst}.out 2> ${dst}.err &"
}

# Instantiate a lmktfy cluster
#
# Assumed vars:
#   LMKTFY_ROOT
#   <Various vars set in config file>
function lmktfy-up {
  verify-ssh-prereqs
  find-release-tars

  ensure-temp-dir

  get-password
  python "${LMKTFY_ROOT}/third_party/htpasswd/htpasswd.py" \
    -b -c "${LMKTFY_TEMP}/htpasswd" "$LMKTFY_USER" "$LMKTFY_PASSWORD"
  local htpasswd
  htpasswd=$(cat "${LMKTFY_TEMP}/htpasswd")

  echo "Starting master VM (this can take a minute)..."

  (
    echo "#! /bin/bash"
    echo "readonly MY_NAME=${MASTER_NAME}"
    grep -v "^#" "${LMKTFY_ROOT}/cluster/vsphere/templates/hostname.sh"
    echo "cd /home/lmktfy/cache/lmktfy-install"
    echo "readonly MASTER_NAME='${MASTER_NAME}'"
    echo "readonly INSTANCE_PREFIX='${INSTANCE_PREFIX}'"
    echo "readonly NODE_INSTANCE_PREFIX='${INSTANCE_PREFIX}-minion'"
    echo "readonly PORTAL_NET='${PORTAL_NET}'"
    echo "readonly ENABLE_NODE_MONITORING='${ENABLE_NODE_MONITORING:-false}'"
    echo "readonly ENABLE_NODE_LOGGING='${ENABLE_NODE_LOGGING:-false}'"
    echo "readonly LOGGING_DESTINATION='${LOGGING_DESTINATION:-}'"
    echo "readonly ENABLE_CLUSTER_DNS='${ENABLE_CLUSTER_DNS:-false}'"
    echo "readonly DNS_SERVER_IP='${DNS_SERVER_IP:-}'"
    echo "readonly DNS_DOMAIN='${DNS_DOMAIN:-}'"
    echo "readonly SERVER_BINARY_TAR='${SERVER_BINARY_TAR##*/}'"
    echo "readonly SALT_TAR='${SALT_TAR##*/}'"
    echo "readonly MASTER_HTPASSWD='${htpasswd}'"
    grep -v "^#" "${LMKTFY_ROOT}/cluster/vsphere/templates/create-dynamic-salt-files.sh"
    grep -v "^#" "${LMKTFY_ROOT}/cluster/vsphere/templates/install-release.sh"
    grep -v "^#" "${LMKTFY_ROOT}/cluster/vsphere/templates/salt-master.sh"
  ) > "${LMKTFY_TEMP}/master-start.sh"

  lmktfy-up-vm ${MASTER_NAME} -c ${MASTER_CPU-1} -m ${MASTER_MEMORY_MB-1024}
  upload-server-tars
  lmktfy-run ${MASTER_NAME} "${LMKTFY_TEMP}/master-start.sh"

  # Print master IP, so user can log in for debugging.
  detect-master
  echo

  echo "Starting minion VMs (this can take a minute)..."

  for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
    (
      echo "#! /bin/bash"
      echo "readonly MY_NAME=${MINION_NAMES[$i]}"
      grep -v "^#" "${LMKTFY_ROOT}/cluster/vsphere/templates/hostname.sh"
      echo "LMKTFY_MASTER=${LMKTFY_MASTER}"
      echo "LMKTFY_MASTER_IP=${LMKTFY_MASTER_IP}"
      echo "MINION_IP_RANGE=${MINION_IP_RANGES[$i]}"
      grep -v "^#" "${LMKTFY_ROOT}/cluster/vsphere/templates/salt-minion.sh"
    ) > "${LMKTFY_TEMP}/minion-start-${i}.sh"

    (
      lmktfy-up-vm "${MINION_NAMES[$i]}" -c ${MINION_CPU-1} -m ${MINION_MEMORY_MB-1024}
      lmktfy-run "${MINION_NAMES[$i]}" "${LMKTFY_TEMP}/minion-start-${i}.sh"
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

  # Print minion IPs, so user can log in for debugging.
  detect-minions
  echo

  echo "Waiting for master and minion initialization."
  echo
  echo "  This will continually check to see if the API for lmktfy is reachable."
  echo "  This might loop forever if there was some uncaught error during start up."
  echo

  printf "Waiting for ${LMKTFY_MASTER} to become available..."
  until curl --insecure --user "${LMKTFY_USER}:${LMKTFY_PASSWORD}" --max-time 5 \
          --fail --output /dev/null --silent "https://${LMKTFY_MASTER_IP}/api/v1beta1/pods"; do
      printf "."
      sleep 2
  done
  printf " OK\n"

  local i
  for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
    printf "Waiting for ${MINION_NAMES[$i]} to become available..."
    until curl --max-time 5 \
            --fail --output /dev/null --silent "http://${LMKTFY_MINION_IP_ADDRESSES[$i]}:10250/healthz"; do
        printf "."
        sleep 2
    done
    printf " OK\n"
  done

  echo
  echo "Sanity checking cluster..."

  sleep 5

  # Basic sanity checking
  local i
  for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
      # Make sure docker is installed
      lmktfy-ssh "${LMKTFY_MINION_IP_ADDRESSES[$i]}" which docker > /dev/null || {
        echo "Docker failed to install on ${MINION_NAMES[$i]}. Your cluster is unlikely" >&2
        echo "to work correctly. Please run ./cluster/lmktfy-down.sh and re-create the" >&2
        echo "cluster. (sorry!)" >&2
        exit 1
      }
  done

  echo
  echo "LMKTFY cluster is running. The master is running at:"
  echo
  echo "  https://${LMKTFY_MASTER_IP}"
  echo
  echo "The user name and password to use is located in ~/.lmktfy_auth."
  echo

  local lmktfy_cert=".lmktfycfg.crt"
  local lmktfy_key=".lmktfycfg.key"
  local ca_cert=".lmktfy.ca.crt"

  (
    umask 077

    lmktfy-ssh "${LMKTFY_MASTER_IP}" sudo cat /srv/lmktfy/lmktfycfg.crt >"${HOME}/${lmktfy_cert}" 2>/dev/null
    lmktfy-ssh "${LMKTFY_MASTER_IP}" sudo cat /srv/lmktfy/lmktfycfg.key >"${HOME}/${lmktfy_key}" 2>/dev/null
    lmktfy-ssh "${LMKTFY_MASTER_IP}" sudo cat /srv/lmktfy/ca.crt >"${HOME}/${ca_cert}" 2>/dev/null

    cat << EOF > ~/.lmktfy_auth
    {
      "User": "$LMKTFY_USER",
      "Password": "$LMKTFY_PASSWORD",
      "CAFile": "$HOME/$ca_cert",
      "CertFile": "$HOME/$lmktfy_cert",
      "KeyFile": "$HOME/$lmktfy_key"
    }
EOF

    chmod 0600 ~/.lmktfy_auth "${HOME}/${lmktfy_cert}" \
      "${HOME}/${lmktfy_key}" "${HOME}/${ca_cert}"
  )
}

# Delete a lmktfy cluster
function lmktfy-down {
  govc vm.destroy ${MASTER_NAME} &

  for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
    govc vm.destroy ${MINION_NAMES[i]} &
  done

  wait
}

# Update a lmktfy cluster with latest source
function lmktfy-push {
  verify-ssh-prereqs
  find-release-tars

  detect-master
  upload-server-tars

  (
    echo "#! /bin/bash"
    echo "cd /home/lmktfy/cache/lmktfy-install"
    echo "readonly SERVER_BINARY_TAR='${SERVER_BINARY_TAR##*/}'"
    echo "readonly SALT_TAR='${SALT_TAR##*/}'"
    grep -v "^#" "${LMKTFY_ROOT}/cluster/vsphere/templates/install-release.sh"
    echo "echo Executing configuration"
    echo "sudo salt '*' mine.update"
    echo "sudo salt --force-color '*' state.highstate"
  ) | lmktfy-ssh "${LMKTFY_MASTER_IP}"

  get-password

  echo
  echo "LMKTFY cluster is running.  The master is running at:"
  echo
  echo "  https://${LMKTFY_MASTER_IP}"
  echo
  echo "The user name and password to use is located in ~/.lmktfy_auth."
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
