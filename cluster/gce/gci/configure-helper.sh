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

# This script is for configuring kubernetes master and node instances. It is
# uploaded in the manifests tar ball.

# TODO: this script duplicates templating logic from cluster/saltbase/salt
# using sed. It should use an actual template parser on the manifest
# files.

set -o errexit
set -o nounset
set -o pipefail

readonly UUID_MNT_PREFIX="/mnt/disks/by-uuid/google-local-ssds"
readonly UUID_BLOCK_PREFIX="/dev/disk/by-uuid/google-local-ssds"

# Use --retry-connrefused opt only if it's supported by curl.
CURL_RETRY_CONNREFUSED=""
if curl --help | grep -q -- '--retry-connrefused'; then
  CURL_RETRY_CONNREFUSED='--retry-connrefused'
fi

function setup-os-params {
  # Reset core_pattern. On GCI, the default core_pattern pipes the core dumps to
  # /sbin/crash_reporter which is more restrictive in saving crash dumps. So for
  # now, set a generic core_pattern that users can work with.
  echo "core.%e.%p.%t" > /proc/sys/kernel/core_pattern
}

function config-ip-firewall {
  echo "Configuring IP firewall rules"

  # Do not consider loopback addresses as martian source or destination while
  # routing. This enables the use of 127/8 for local routing purposes.
  sysctl -w net.ipv4.conf.all.route_localnet=1

  # The GCI image has host firewall which drop most inbound/forwarded packets.
  # We need to add rules to accept all TCP/UDP/ICMP packets.
  if iptables -w -L INPUT | grep "Chain INPUT (policy DROP)" > /dev/null; then
    echo "Add rules to accept all inbound TCP/UDP/ICMP packets"
    iptables -A INPUT -w -p TCP -j ACCEPT
    iptables -A INPUT -w -p UDP -j ACCEPT
    iptables -A INPUT -w -p ICMP -j ACCEPT
  fi
  if iptables -w -L FORWARD | grep "Chain FORWARD (policy DROP)" > /dev/null; then
    echo "Add rules to accept all forwarded TCP/UDP/ICMP packets"
    iptables -A FORWARD -w -p TCP -j ACCEPT
    iptables -A FORWARD -w -p UDP -j ACCEPT
    iptables -A FORWARD -w -p ICMP -j ACCEPT
  fi

  # Flush iptables nat table
  iptables -w -t nat -F || true

  if [[ "${NON_MASQUERADE_CIDR:-}" == "0.0.0.0/0" ]]; then
    echo "Add rules for ip masquerade"
    iptables -w -t nat -N IP-MASQ
    iptables -w -t nat -A POSTROUTING -m comment --comment "ip-masq: ensure nat POSTROUTING directs all non-LOCAL destination traffic to our custom IP-MASQ chain" -m addrtype ! --dst-type LOCAL -j IP-MASQ
    iptables -w -t nat -A IP-MASQ -d 169.254.0.0/16 -m comment --comment "ip-masq: local traffic is not subject to MASQUERADE" -j RETURN
    iptables -w -t nat -A IP-MASQ -d 10.0.0.0/8 -m comment --comment "ip-masq: local traffic is not subject to MASQUERADE" -j RETURN
    iptables -w -t nat -A IP-MASQ -d 172.16.0.0/12 -m comment --comment "ip-masq: local traffic is not subject to MASQUERADE" -j RETURN
    iptables -w -t nat -A IP-MASQ -d 192.168.0.0/16 -m comment --comment "ip-masq: local traffic is not subject to MASQUERADE" -j RETURN
    iptables -w -t nat -A IP-MASQ -m comment --comment "ip-masq: outbound traffic is subject to MASQUERADE (must be last in chain)" -j MASQUERADE
  fi

  # If METADATA_CONCEALMENT_NO_FIREWALL is set, don't create a firewall on this
  # node because we don't expect the daemonset to run on this node.
  if [[ "${ENABLE_METADATA_CONCEALMENT:-}" == "true" ]] && [[ ! "${METADATA_CONCEALMENT_NO_FIREWALL:-}" == "true" ]]; then
    echo "Add rule for metadata concealment"
    iptables -w -t nat -I PREROUTING -p tcp -d 169.254.169.254 --dport 80 -m comment --comment "metadata-concealment: bridge traffic to metadata server goes to metadata proxy" -j DNAT --to-destination 127.0.0.1:988
  fi
}

function create-dirs {
  echo "Creating required directories"
  mkdir -p /var/lib/kubelet
  mkdir -p /etc/kubernetes/manifests
  if [[ "${KUBERNETES_MASTER:-}" == "false" ]]; then
    mkdir -p /var/lib/kube-proxy
  fi
}

# Gets the total number of $(1) and $(2) type disks specified
# by the user in ${NODE_LOCAL_SSDS_EXT}
function get-local-disk-num() {
  local interface="${1}"
  local format="${2}"

  localdisknum=0
  if [[ ! -z "${NODE_LOCAL_SSDS_EXT:-}" ]]; then
    IFS=";" read -r -a ssdgroups <<< "${NODE_LOCAL_SSDS_EXT:-}"
    for ssdgroup in "${ssdgroups[@]}"; do
      IFS="," read -r -a ssdopts <<< "${ssdgroup}"
      local opnum="${ssdopts[0]}"
      local opinterface="${ssdopts[1]}"
      local opformat="${ssdopts[2]}"

      if [[ "${opformat,,}" == "${format,,}" && "${opinterface,,}" == "${interface,,}" ]]; then
        localdisknum=$((localdisknum+opnum))
      fi
    done
  fi
}

# Creates a symlink for a ($1) so that it may be used as block storage
function safe-block-symlink(){
  local device="${1}"
  local symdir="${2}"
  
  mkdir -p "${symdir}"

  get-or-generate-uuid "${device}"
  local myuuid="${retuuid}"

  local sym="${symdir}/local-ssd-${myuuid}"
  # Do not "mkdir -p ${sym}" as that will cause unintended symlink behavior
  ln -s "${device}" "${sym}"
  echo "Created a symlink for SSD $ssd at ${sym}"
  chmod a+w "${sym}"
}

# Gets a pregenerated UUID from ${ssdmap} if it exists, otherwise generates a new
# UUID and places it inside ${ssdmap}
function get-or-generate-uuid(){
  local device="${1}"

  local ssdmap="/home/kubernetes/localssdmap.txt"
  echo "Generating or getting UUID from ${ssdmap}"

  if [[ ! -e "${ssdmap}" ]]; then
    touch "${ssdmap}"
    chmod +w "${ssdmap}"
  fi

  # each line of the ssdmap looks like "${device} persistent-uuid"
  if [[ ! -z $(grep ${device} ${ssdmap}) ]]; then
    #create symlink based on saved uuid
    local myuuid=$(grep ${device} ${ssdmap} | cut -d ' ' -f 2)
  else
    # generate new uuid and add it to the map
    local myuuid=$(uuidgen)
    if [[ ! ${?} -eq 0 ]]; then
      echo "Failed to generate valid UUID with uuidgen" >&2
      exit 2
    fi
    echo "${device} ${myuuid}" >> "${ssdmap}"
  fi

  if [[ -z "${myuuid}" ]]; then
    echo "Failed to get a uuid for device ${device} when symlinking." >&2
    exit 2
  fi

  retuuid="${myuuid}"
}

#Formats the given device ($1) if needed and mounts it at given mount point
# ($2).
function safe-format-and-mount() {
  local device="${1}"
  local mountpoint="${2}"

  # Format only if the disk is not already formatted.
  if ! tune2fs -l "${device}" ; then
    echo "Formatting '${device}'"
    mkfs.ext4 -F "${device}"
  fi

  mkdir -p "${mountpoint}"
  echo "Mounting '${device}' at '${mountpoint}'"
  mount -o discard,defaults "${device}" "${mountpoint}"
}

# Gets a devices UUID and bind mounts the device to mount location in
# /mnt/disks/by-id/
function unique-uuid-bind-mount(){
  local mountpoint="${1}"
  local actual_device="${2}"

  # Trigger udev refresh so that newly formatted devices are propagated in by-uuid
  udevadm control --reload-rules
  udevadm trigger
  udevadm settle 

  # grep the exact match of actual device, prevents substring matching
  local myuuid=$(ls -l /dev/disk/by-uuid/ | grep "/${actual_device}$" | tr -s ' ' | cut -d ' ' -f 9)
  # myuuid should be the uuid of the device as found in /dev/disk/by-uuid/ 
  if [[ -z "${myuuid}" ]]; then
    echo "Failed to get a uuid for device ${actual_device} when mounting." >&2
    exit 2
  fi

  # bindpoint should be the full path of the to-be-bound device
  local bindpoint="${UUID_MNT_PREFIX}-${interface}-fs/local-ssd-${myuuid}"

  safe-bind-mount "${mountpoint}" "${bindpoint}"
}

# Bind mounts device at mountpoint to bindpoint
function safe-bind-mount(){
  local mountpoint="${1}"
  local bindpoint="${2}"

  # Mount device to the mountpoint
  mkdir -p "${bindpoint}"
  echo "Binding '${mountpoint}' at '${bindpoint}'"
  mount --bind "${mountpoint}" "${bindpoint}"
  chmod a+w "${bindpoint}"
}


# Mounts, bindmounts, or symlinks depending on the interface and format
# of the incoming device
function mount-ext(){
  local ssd="${1}"
  local devicenum="${2}"
  local interface="${3}"
  local format="${4}"
  

  if [[ -z "${devicenum}" ]]; then
    echo "Failed to get the local disk number for device ${ssd}" >&2
    exit 2
  fi

  # TODO: Handle partitioned disks. Right now this code just ignores partitions
  if [[ "${format}" == "fs" ]]; then
    if [[ "${interface}" == "scsi" ]]; then
      local actual_device=$(readlink -f "${ssd}" | cut -d '/' -f 3)
      # Error checking
      if [[ "${actual_device}" != sd* ]]; then
        echo "'actual_device' is not of the correct format. It must be the kernel name of the device, got ${actual_device} instead" >&2
        exit 1
      fi
      local mountpoint="/mnt/disks/ssd${devicenum}"
    else
      # This path is required because the existing Google images do not
      # expose NVMe devices in /dev/disk/by-id so we are using the /dev/nvme instead
      local actual_device=$(echo ${ssd} | cut -d '/' -f 3)
      # Error checking
      if [[ "${actual_device}" != nvme* ]]; then
        echo "'actual_device' is not of the correct format. It must be the kernel name of the device, got ${actual_device} instead" >&2
        exit 1
      fi
      local mountpoint="/mnt/disks/ssd-nvme${devicenum}"
    fi

    safe-format-and-mount "${ssd}" "${mountpoint}"
    # We only do the bindmount if users are using the new local ssd request method
    # see https://github.com/kubernetes/kubernetes/pull/53466#discussion_r146431894
    if [[ ! -z "${NODE_LOCAL_SSDS_EXT:-}" ]]; then
      unique-uuid-bind-mount "${mountpoint}" "${actual_device}"
    fi
  elif [[ "${format}" == "block" ]]; then
    local symdir="${UUID_BLOCK_PREFIX}-${interface}-block"
    safe-block-symlink "${ssd}" "${symdir}"
  else
    echo "Disk format must be either fs or block, got ${format}"
  fi
}

# Local ssds, if present, are mounted or symlinked to their appropriate
# locations
function ensure-local-ssds() {
  get-local-disk-num "scsi" "block"
  local scsiblocknum="${localdisknum}"
  local i=0
  for ssd in /dev/disk/by-id/google-local-ssd-*; do
    if [ -e "${ssd}" ]; then
      local devicenum=`echo ${ssd} | sed -e 's/\/dev\/disk\/by-id\/google-local-ssd-\([0-9]*\)/\1/'`
      if [[ "${i}" -lt "${scsiblocknum}" ]]; then
        mount-ext "${ssd}" "${devicenum}" "scsi" "block"
      else
        # GKE does not set NODE_LOCAL_SSDS so all non-block devices
        # are assumed to be filesystem devices
        mount-ext "${ssd}" "${devicenum}" "scsi" "fs"
      fi
      i=$((i+1))
    else
      echo "No local SCSI SSD disks found."
    fi
  done

  # The following mounts or symlinks NVMe devices
  get-local-disk-num "nvme" "block"
  local nvmeblocknum="${localdisknum}"
  local i=0
  for ssd in /dev/nvme*; do
    if [ -e "${ssd}" ]; then
      # This workaround to find if the NVMe device is a disk is required because
      # the existing Google images does not expose NVMe devices in /dev/disk/by-id
      if [[ `udevadm info --query=property --name=${ssd} | grep DEVTYPE | sed "s/DEVTYPE=//"` == "disk" ]]; then
        local devicenum=`echo ${ssd} | sed -e 's/\/dev\/nvme0n\([0-9]*\)/\1/'`
        if [[ "${i}" -lt "${nvmeblocknum}" ]]; then
          mount-ext "${ssd}" "${devicenum}" "nvme" "block"
        else
          mount-ext "${ssd}" "${devicenum}" "nvme" "fs"
        fi
        i=$((i+1))
      fi
    else
      echo "No local NVMe SSD disks found."
    fi
  done
}

# Installs logrotate configuration files
function setup-logrotate() {
  mkdir -p /etc/logrotate.d/
  # Configure log rotation for all logs in /var/log, which is where k8s services
  # are configured to write their log files. Whenever logrotate is ran, this
  # config will:
  # * rotate the log file if its size is > 100Mb OR if one day has elapsed
  # * save rotated logs into a gzipped timestamped backup
  # * log file timestamp (controlled by 'dateformat') includes seconds too. This
  #   ensures that logrotate can generate unique logfiles during each rotation
  #   (otherwise it skips rotation if 'maxsize' is reached multiple times in a
  #   day).
  # * keep only 5 old (rotated) logs, and will discard older logs.
  cat > /etc/logrotate.d/allvarlogs <<EOF
/var/log/*.log {
    rotate ${LOGROTATE_FILES_MAX_COUNT:-5}
    copytruncate
    missingok
    notifempty
    compress
    maxsize ${LOGROTATE_MAX_SIZE:-100M}
    daily
    dateext
    dateformat -%Y%m%d-%s
    create 0644 root root
}
EOF

}

# Finds the master PD device; returns it in MASTER_PD_DEVICE
function find-master-pd {
  MASTER_PD_DEVICE=""
  if [[ ! -e /dev/disk/by-id/google-master-pd ]]; then
    return
  fi
  device_info=$(ls -l /dev/disk/by-id/google-master-pd)
  relative_path=${device_info##* }
  MASTER_PD_DEVICE="/dev/disk/by-id/${relative_path}"
}

# Mounts a persistent disk (formatting if needed) to store the persistent data
# on the master -- etcd's data, a few settings, and security certs/keys/tokens.
# safe-format-and-mount only formats an unformatted disk, and mkdir -p will
# leave a directory be if it already exists.
function mount-master-pd {
  find-master-pd
  if [[ -z "${MASTER_PD_DEVICE:-}" ]]; then
    return
  fi

  echo "Mounting master-pd"
  local -r pd_path="/dev/disk/by-id/google-master-pd"
  local -r mount_point="/mnt/disks/master-pd"
  # Format and mount the disk, create directories on it for all of the master's
  # persistent data, and link them to where they're used.
  mkdir -p "${mount_point}"
  safe-format-and-mount "${pd_path}" "${mount_point}"
  echo "Mounted master-pd '${pd_path}' at '${mount_point}'"

  # NOTE: These locations on the PD store persistent data, so to maintain
  # upgradeability, these locations should not change.  If they do, take care
  # to maintain a migration path from these locations to whatever new
  # locations.

  # Contains all the data stored in etcd.
  mkdir -m 700 -p "${mount_point}/var/etcd"
  ln -s -f "${mount_point}/var/etcd" /var/etcd
  mkdir -p /etc/srv
  # Contains the dynamically generated apiserver auth certs and keys.
  mkdir -p "${mount_point}/srv/kubernetes"
  ln -s -f "${mount_point}/srv/kubernetes" /etc/srv/kubernetes
  # Directory for kube-apiserver to store SSH key (if necessary).
  mkdir -p "${mount_point}/srv/sshproxy"
  ln -s -f "${mount_point}/srv/sshproxy" /etc/srv/sshproxy

  if ! id etcd &>/dev/null; then
    useradd -s /sbin/nologin -d /var/etcd etcd
  fi
  chown -R etcd "${mount_point}/var/etcd"
  chgrp -R etcd "${mount_point}/var/etcd"
}

# append_or_replace_prefixed_line ensures:
# 1. the specified file exists
# 2. existing lines with the specified ${prefix} are removed
# 3. a new line with the specified ${prefix}${suffix} is appended
function append_or_replace_prefixed_line {
  local -r file="${1:-}"
  local -r prefix="${2:-}"
  local -r suffix="${3:-}"
  local -r dirname="$(dirname ${file})"
  local -r tmpfile="$(mktemp -t filtered.XXXX --tmpdir=${dirname})"

  touch "${file}"
  awk "substr(\$0,0,length(\"${prefix}\")) != \"${prefix}\" { print }" "${file}" > "${tmpfile}"
  echo "${prefix}${suffix}" >> "${tmpfile}"
  mv "${tmpfile}" "${file}"
}

function write-pki-data {
  local data="${1}"
  local path="${2}"
  (umask 077; echo "${data}" | base64 --decode > "${path}")
}

function create-node-pki {
  echo "Creating node pki files"

  local -r pki_dir="/etc/srv/kubernetes/pki"
  mkdir -p "${pki_dir}"

  if [[ -z "${CA_CERT_BUNDLE:-}" ]]; then
    CA_CERT_BUNDLE="${CA_CERT}"
  fi

  CA_CERT_BUNDLE_PATH="${pki_dir}/ca-certificates.crt"
  write-pki-data "${CA_CERT_BUNDLE}" "${CA_CERT_BUNDLE_PATH}"

  if [[ ! -z "${KUBELET_CERT:-}" && ! -z "${KUBELET_KEY:-}" ]]; then
    KUBELET_CERT_PATH="${pki_dir}/kubelet.crt"
    write-pki-data "${KUBELET_CERT}" "${KUBELET_CERT_PATH}"

    KUBELET_KEY_PATH="${pki_dir}/kubelet.key"
    write-pki-data "${KUBELET_KEY}" "${KUBELET_KEY_PATH}"
  fi
}

function create-master-pki {
  echo "Creating master pki files"

  local -r pki_dir="/etc/srv/kubernetes/pki"
  mkdir -p "${pki_dir}"

  CA_CERT_PATH="${pki_dir}/ca.crt"
  write-pki-data "${CA_CERT}" "${CA_CERT_PATH}"

  # this is not true on GKE
  if [[ ! -z "${CA_KEY:-}" ]]; then
    CA_KEY_PATH="${pki_dir}/ca.key"
    write-pki-data "${CA_KEY}" "${CA_KEY_PATH}"
  fi

  if [[ -z "${APISERVER_SERVER_CERT:-}" || -z "${APISERVER_SERVER_KEY:-}" ]]; then
    APISERVER_SERVER_CERT="${MASTER_CERT}"
    APISERVER_SERVER_KEY="${MASTER_KEY}"
  fi

  APISERVER_SERVER_CERT_PATH="${pki_dir}/apiserver.crt"
  write-pki-data "${APISERVER_SERVER_CERT}" "${APISERVER_SERVER_CERT_PATH}"

  APISERVER_SERVER_KEY_PATH="${pki_dir}/apiserver.key"
  write-pki-data "${APISERVER_SERVER_KEY}" "${APISERVER_SERVER_KEY_PATH}"

  if [[ -z "${APISERVER_CLIENT_CERT:-}" || -z "${APISERVER_CLIENT_KEY:-}" ]]; then
    APISERVER_CLIENT_CERT="${KUBEAPISERVER_CERT}"
    APISERVER_CLIENT_KEY="${KUBEAPISERVER_KEY}"
  fi

  APISERVER_CLIENT_CERT_PATH="${pki_dir}/apiserver-client.crt"
  write-pki-data "${APISERVER_CLIENT_CERT}" "${APISERVER_CLIENT_CERT_PATH}"

  APISERVER_CLIENT_KEY_PATH="${pki_dir}/apiserver-client.key"
  write-pki-data "${APISERVER_CLIENT_KEY}" "${APISERVER_CLIENT_KEY_PATH}"

  if [[ -z "${SERVICEACCOUNT_CERT:-}" || -z "${SERVICEACCOUNT_KEY:-}" ]]; then
    SERVICEACCOUNT_CERT="${MASTER_CERT}"
    SERVICEACCOUNT_KEY="${MASTER_KEY}"
  fi

  SERVICEACCOUNT_CERT_PATH="${pki_dir}/serviceaccount.crt"
  write-pki-data "${SERVICEACCOUNT_CERT}" "${SERVICEACCOUNT_CERT_PATH}"

  SERVICEACCOUNT_KEY_PATH="${pki_dir}/serviceaccount.key"
  write-pki-data "${SERVICEACCOUNT_KEY}" "${SERVICEACCOUNT_KEY_PATH}"

  if [[ ! -z "${REQUESTHEADER_CA_CERT:-}" ]]; then
    AGGREGATOR_CA_KEY_PATH="${pki_dir}/aggr_ca.key"
    write-pki-data "${AGGREGATOR_CA_KEY}" "${AGGREGATOR_CA_KEY_PATH}"

    REQUESTHEADER_CA_CERT_PATH="${pki_dir}/aggr_ca.crt"
    write-pki-data "${REQUESTHEADER_CA_CERT}" "${REQUESTHEADER_CA_CERT_PATH}"

    PROXY_CLIENT_KEY_PATH="${pki_dir}/proxy_client.key"
    write-pki-data "${PROXY_CLIENT_KEY}" "${PROXY_CLIENT_KEY_PATH}"

    PROXY_CLIENT_CERT_PATH="${pki_dir}/proxy_client.crt"
    write-pki-data "${PROXY_CLIENT_CERT}" "${PROXY_CLIENT_CERT_PATH}"
  fi
}

# After the first boot and on upgrade, these files exist on the master-pd
# and should never be touched again (except perhaps an additional service
# account, see NB below.) One exception is if METADATA_CLOBBERS_CONFIG is
# enabled. In that case the basic_auth.csv file will be rewritten to make
# sure it matches the metadata source of truth.
function create-master-auth {
  echo "Creating master auth files"
  local -r auth_dir="/etc/srv/kubernetes"
  local -r basic_auth_csv="${auth_dir}/basic_auth.csv"
  if [[ -n "${KUBE_PASSWORD:-}" && -n "${KUBE_USER:-}" ]]; then
    if [[ -e "${basic_auth_csv}" && "${METADATA_CLOBBERS_CONFIG:-false}" == "true" ]]; then
      # If METADATA_CLOBBERS_CONFIG is true, we want to rewrite the file
      # completely, because if we're changing KUBE_USER and KUBE_PASSWORD, we
      # have nothing to match on.  The file is replaced just below with
      # append_or_replace_prefixed_line.
      rm "${basic_auth_csv}"
    fi
    append_or_replace_prefixed_line "${basic_auth_csv}" "${KUBE_PASSWORD},${KUBE_USER},"      "admin,system:masters"
  fi

  local -r known_tokens_csv="${auth_dir}/known_tokens.csv"
  if [[ -e "${known_tokens_csv}" && "${METADATA_CLOBBERS_CONFIG:-false}" == "true" ]]; then
    rm "${known_tokens_csv}"
  fi
  if [[ -n "${KUBE_BEARER_TOKEN:-}" ]]; then
    append_or_replace_prefixed_line "${known_tokens_csv}" "${KUBE_BEARER_TOKEN},"             "admin,admin,system:masters"
  fi
  if [[ -n "${KUBE_CONTROLLER_MANAGER_TOKEN:-}" ]]; then
    append_or_replace_prefixed_line "${known_tokens_csv}" "${KUBE_CONTROLLER_MANAGER_TOKEN}," "system:kube-controller-manager,uid:system:kube-controller-manager"
  fi
  if [[ -n "${KUBE_SCHEDULER_TOKEN:-}" ]]; then
    append_or_replace_prefixed_line "${known_tokens_csv}" "${KUBE_SCHEDULER_TOKEN},"          "system:kube-scheduler,uid:system:kube-scheduler"
  fi
  if [[ -n "${KUBE_PROXY_TOKEN:-}" ]]; then
    append_or_replace_prefixed_line "${known_tokens_csv}" "${KUBE_PROXY_TOKEN},"              "system:kube-proxy,uid:kube_proxy"
  fi
  if [[ -n "${NODE_PROBLEM_DETECTOR_TOKEN:-}" ]]; then
    append_or_replace_prefixed_line "${known_tokens_csv}" "${NODE_PROBLEM_DETECTOR_TOKEN},"   "system:node-problem-detector,uid:node-problem-detector"
  fi
  local use_cloud_config="false"
  cat <<EOF >/etc/gce.conf
[global]
EOF
  if [[ -n "${GCE_API_ENDPOINT:-}" ]]; then
    cat <<EOF >>/etc/gce.conf
api-endpoint = ${GCE_API_ENDPOINT}
EOF
  fi
  if [[ -n "${TOKEN_URL:-}" && -n "${TOKEN_BODY:-}" ]]; then
    use_cloud_config="true"
    cat <<EOF >>/etc/gce.conf
token-url = ${TOKEN_URL}
token-body = ${TOKEN_BODY}
EOF
  fi
  if [[ -n "${PROJECT_ID:-}" ]]; then
    use_cloud_config="true"
    cat <<EOF >>/etc/gce.conf
project-id = ${PROJECT_ID}
EOF
  fi
  if [[ -n "${NETWORK_PROJECT_ID:-}" ]]; then
    use_cloud_config="true"
    cat <<EOF >>/etc/gce.conf
network-project-id = ${NETWORK_PROJECT_ID}
EOF
  fi
  if [[ -n "${NODE_NETWORK:-}" ]]; then
    use_cloud_config="true"
    cat <<EOF >>/etc/gce.conf
network-name = ${NODE_NETWORK}
EOF
  fi
  if [[ -n "${NODE_SUBNETWORK:-}" ]]; then
    use_cloud_config="true"
    cat <<EOF >>/etc/gce.conf
subnetwork-name = ${NODE_SUBNETWORK}
EOF
  fi
  if [[ -n "${NODE_INSTANCE_PREFIX:-}" ]]; then
    use_cloud_config="true"
    if [[ -n "${NODE_TAGS:-}" ]]; then
      # split NODE_TAGS into an array by comma.
      IFS=',' read -r -a node_tags <<< ${NODE_TAGS}
    else
      local -r node_tags="${NODE_INSTANCE_PREFIX}"
    fi
    cat <<EOF >>/etc/gce.conf
node-instance-prefix = ${NODE_INSTANCE_PREFIX}
EOF
    for tag in ${node_tags[@]}; do
      cat <<EOF >>/etc/gce.conf
node-tags = ${tag}
EOF
    done
  fi
  if [[ -n "${MULTIZONE:-}" ]]; then
    use_cloud_config="true"
    cat <<EOF >>/etc/gce.conf
multizone = ${MULTIZONE}
EOF
  fi
  if [[ -n "${GCE_ALPHA_FEATURES:-}" ]]; then
    use_cloud_config="true"
    # split GCE_ALPHA_FEATURES into an array by comma.
    IFS=',' read -r -a alpha_features <<< ${GCE_ALPHA_FEATURES}
    for feature in ${alpha_features[@]}; do
      cat <<EOF >>/etc/gce.conf
alpha-features = ${feature}
EOF
    done
  fi
  if [[ -n "${SECONDARY_RANGE_NAME:-}" ]]; then
    use_cloud_config="true"
    cat <<EOF >> /etc/gce.conf
secondary-range-name = ${SECONDARY_RANGE_NAME}
EOF
  fi
  if [[ "${use_cloud_config}" != "true" ]]; then
    rm -f /etc/gce.conf
  fi

  if [[ -n "${GCP_AUTHN_URL:-}" ]]; then
    cat <<EOF >/etc/gcp_authn.config
clusters:
  - name: gcp-authentication-server
    cluster:
      server: ${GCP_AUTHN_URL}
users:
  - name: kube-apiserver
    user:
      auth-provider:
        name: gcp
current-context: webhook
contexts:
- context:
    cluster: gcp-authentication-server
    user: kube-apiserver
  name: webhook
EOF
  fi

  if [[ -n "${GCP_AUTHZ_URL:-}" ]]; then
    cat <<EOF >/etc/gcp_authz.config
clusters:
  - name: gcp-authorization-server
    cluster:
      server: ${GCP_AUTHZ_URL}
users:
  - name: kube-apiserver
    user:
      auth-provider:
        name: gcp
current-context: webhook
contexts:
- context:
    cluster: gcp-authorization-server
    user: kube-apiserver
  name: webhook
EOF
  fi

if [[ -n "${GCP_IMAGE_VERIFICATION_URL:-}" ]]; then
    # This is the config file for the image review webhook.
    cat <<EOF >/etc/gcp_image_review.config
clusters:
  - name: gcp-image-review-server
    cluster:
      server: ${GCP_IMAGE_VERIFICATION_URL}
users:
  - name: kube-apiserver
    user:
      auth-provider:
        name: gcp
current-context: webhook
contexts:
- context:
    cluster: gcp-image-review-server
    user: kube-apiserver
  name: webhook
EOF
    # This is the config for the image review admission controller.
    cat <<EOF >/etc/admission_controller.config
imagePolicy:
  kubeConfigFile: /etc/gcp_image_review.config
  allowTTL: 30
  denyTTL: 30
  retryBackoff: 500
  defaultAllow: true
EOF
  fi
}

# Write the config for the audit policy.
function create-master-audit-policy {
  local -r path="${1}"
  local -r policy="${2:-}"

  if [[ -n "${policy}" ]]; then
    echo "${policy}" > "${path}"
    return
  fi

  # Known api groups
  local -r known_apis='
      - group: "" # core
      - group: "admissionregistration.k8s.io"
      - group: "apiextensions.k8s.io"
      - group: "apiregistration.k8s.io"
      - group: "apps"
      - group: "authentication.k8s.io"
      - group: "authorization.k8s.io"
      - group: "autoscaling"
      - group: "batch"
      - group: "certificates.k8s.io"
      - group: "extensions"
      - group: "metrics.k8s.io"
      - group: "networking.k8s.io"
      - group: "policy"
      - group: "rbac.authorization.k8s.io"
      - group: "settings.k8s.io"
      - group: "storage.k8s.io"'

  cat <<EOF >"${path}"
apiVersion: audit.k8s.io/v1beta1
kind: Policy
rules:
  # The following requests were manually identified as high-volume and low-risk,
  # so drop them.
  - level: None
    users: ["system:kube-proxy"]
    verbs: ["watch"]
    resources:
      - group: "" # core
        resources: ["endpoints", "services", "services/status"]
  - level: None
    # Ingress controller reads 'configmaps/ingress-uid' through the unsecured port.
    # TODO(#46983): Change this to the ingress controller service account.
    users: ["system:unsecured"]
    namespaces: ["kube-system"]
    verbs: ["get"]
    resources:
      - group: "" # core
        resources: ["configmaps"]
  - level: None
    users: ["kubelet"] # legacy kubelet identity
    verbs: ["get"]
    resources:
      - group: "" # core
        resources: ["nodes", "nodes/status"]
  - level: None
    userGroups: ["system:nodes"]
    verbs: ["get"]
    resources:
      - group: "" # core
        resources: ["nodes", "nodes/status"]
  - level: None
    users:
      - system:kube-controller-manager
      - system:kube-scheduler
      - system:serviceaccount:kube-system:endpoint-controller
    verbs: ["get", "update"]
    namespaces: ["kube-system"]
    resources:
      - group: "" # core
        resources: ["endpoints"]
  - level: None
    users: ["system:apiserver"]
    verbs: ["get"]
    resources:
      - group: "" # core
        resources: ["namespaces", "namespaces/status", "namespaces/finalize"]
  # Don't log HPA fetching metrics.
  - level: None
    users:
      - system:kube-controller-manager
    verbs: ["get", "list"]
    resources:
      - group: "metrics.k8s.io"

  # Don't log these read-only URLs.
  - level: None
    nonResourceURLs:
      - /healthz*
      - /version
      - /swagger*

  # Don't log events requests.
  - level: None
    resources:
      - group: "" # core
        resources: ["events"]

  # node and pod status calls from nodes are high-volume and can be large, don't log responses for expected updates from nodes
  - level: Request
    users: ["kubelet", "system:node-problem-detector", "system:serviceaccount:kube-system:node-problem-detector"]
    verbs: ["update","patch"]
    resources:
      - group: "" # core
        resources: ["nodes/status", "pods/status"]
    omitStages:
      - "RequestReceived"
  - level: Request
    userGroups: ["system:nodes"]
    verbs: ["update","patch"]
    resources:
      - group: "" # core
        resources: ["nodes/status", "pods/status"]
    omitStages:
      - "RequestReceived"

  # deletecollection calls can be large, don't log responses for expected namespace deletions
  - level: Request
    users: ["system:serviceaccount:kube-system:namespace-controller"]
    verbs: ["deletecollection"]
    omitStages:
      - "RequestReceived"

  # Secrets, ConfigMaps, and TokenReviews can contain sensitive & binary data,
  # so only log at the Metadata level.
  - level: Metadata
    resources:
      - group: "" # core
        resources: ["secrets", "configmaps"]
      - group: authentication.k8s.io
        resources: ["tokenreviews"]
    omitStages:
      - "RequestReceived"
  # Get repsonses can be large; skip them.
  - level: Request
    verbs: ["get", "list", "watch"]
    resources: ${known_apis}
    omitStages:
      - "RequestReceived"
  # Default level for known APIs
  - level: RequestResponse
    resources: ${known_apis}
    omitStages:
      - "RequestReceived"
  # Default level for all other requests.
  - level: Metadata
    omitStages:
      - "RequestReceived"
EOF
}

# Writes the configuration file used by the webhook advanced auditing backend.
function create-master-audit-webhook-config {
  local -r path="${1}"

  if [[ -n "${GCP_AUDIT_URL:-}" ]]; then
    # The webhook config file is a kubeconfig file describing the webhook endpoint.
    cat <<EOF >"${path}"
clusters:
  - name: gcp-audit-server
    cluster:
      server: ${GCP_AUDIT_URL}
users:
  - name: kube-apiserver
    user:
      auth-provider:
        name: gcp
current-context: webhook
contexts:
- context:
    cluster: gcp-audit-server
    user: kube-apiserver
  name: webhook
EOF
  fi
}

# Arg 1: the IP address of the API server
function create-kubelet-kubeconfig() {
  local apiserver_address="${1}"
  if [[ -z "${apiserver_address}" ]]; then
    echo "Must provide API server address to create Kubelet kubeconfig file!"
    exit 1
  fi
  echo "Creating kubelet kubeconfig file"
  cat <<EOF >/var/lib/kubelet/bootstrap-kubeconfig
apiVersion: v1
kind: Config
users:
- name: kubelet
  user:
    client-certificate: ${KUBELET_CERT_PATH}
    client-key: ${KUBELET_KEY_PATH}
clusters:
- name: local
  cluster:
    server: https://${apiserver_address}
    certificate-authority: ${CA_CERT_BUNDLE_PATH}
contexts:
- context:
    cluster: local
    user: kubelet
  name: service-account-context
current-context: service-account-context
EOF
}

# Uses KUBELET_CA_CERT (falling back to CA_CERT), KUBELET_CERT, and KUBELET_KEY
# to generate a kubeconfig file for the kubelet to securely connect to the apiserver.
# Set REGISTER_MASTER_KUBELET to true if kubelet on the master node
# should register to the apiserver.
function create-master-kubelet-auth {
  # Only configure the kubelet on the master if the required variables are
  # set in the environment.
  if [[ -n "${KUBELET_APISERVER:-}" && -n "${KUBELET_CERT:-}" && -n "${KUBELET_KEY:-}" ]]; then
    REGISTER_MASTER_KUBELET="true"
    create-kubelet-kubeconfig ${KUBELET_APISERVER}
  fi
}

function create-kubeproxy-user-kubeconfig {
  echo "Creating kube-proxy user kubeconfig file"
  cat <<EOF >/var/lib/kube-proxy/kubeconfig
apiVersion: v1
kind: Config
users:
- name: kube-proxy
  user:
    token: ${KUBE_PROXY_TOKEN}
clusters:
- name: local
  cluster:
    certificate-authority-data: ${CA_CERT_BUNDLE}
contexts:
- context:
    cluster: local
    user: kube-proxy
  name: service-account-context
current-context: service-account-context
EOF
}

function create-kubecontrollermanager-kubeconfig {
  echo "Creating kube-controller-manager kubeconfig file"
  mkdir -p /etc/srv/kubernetes/kube-controller-manager
  cat <<EOF >/etc/srv/kubernetes/kube-controller-manager/kubeconfig
apiVersion: v1
kind: Config
users:
- name: kube-controller-manager
  user:
    token: ${KUBE_CONTROLLER_MANAGER_TOKEN}
clusters:
- name: local
  cluster:
    insecure-skip-tls-verify: true
    server: https://localhost:443
contexts:
- context:
    cluster: local
    user: kube-controller-manager
  name: service-account-context
current-context: service-account-context
EOF
}

function create-kubescheduler-kubeconfig {
  echo "Creating kube-scheduler kubeconfig file"
  mkdir -p /etc/srv/kubernetes/kube-scheduler
  cat <<EOF >/etc/srv/kubernetes/kube-scheduler/kubeconfig
apiVersion: v1
kind: Config
users:
- name: kube-scheduler
  user:
    token: ${KUBE_SCHEDULER_TOKEN}
clusters:
- name: local
  cluster:
    insecure-skip-tls-verify: true
    server: https://localhost:443
contexts:
- context:
    cluster: local
    user: kube-scheduler
  name: kube-scheduler
current-context: kube-scheduler
EOF
}

function create-kubescheduler-policy-config {
  echo "Creating kube-scheduler policy config file"
  mkdir -p /etc/srv/kubernetes/kube-scheduler
  cat <<EOF >/etc/srv/kubernetes/kube-scheduler/policy-config
${SCHEDULER_POLICY_CONFIG}
EOF
}

function create-node-problem-detector-kubeconfig {
  echo "Creating node-problem-detector kubeconfig file"
  mkdir -p /var/lib/node-problem-detector
  cat <<EOF >/var/lib/node-problem-detector/kubeconfig
apiVersion: v1
kind: Config
users:
- name: node-problem-detector
  user:
    token: ${NODE_PROBLEM_DETECTOR_TOKEN}
clusters:
- name: local
  cluster:
    certificate-authority-data: ${CA_CERT}
contexts:
- context:
    cluster: local
    user: node-problem-detector
  name: service-account-context
current-context: service-account-context
EOF
}

function create-master-etcd-auth {
  if [[ -n "${ETCD_CA_CERT:-}" && -n "${ETCD_PEER_KEY:-}" && -n "${ETCD_PEER_CERT:-}" ]]; then
    local -r auth_dir="/etc/srv/kubernetes"
    echo "${ETCD_CA_CERT}" | base64 --decode | gunzip > "${auth_dir}/etcd-ca.crt"
    echo "${ETCD_PEER_KEY}" | base64 --decode > "${auth_dir}/etcd-peer.key"
    echo "${ETCD_PEER_CERT}" | base64 --decode | gunzip > "${auth_dir}/etcd-peer.crt"
  fi
}

function assemble-docker-flags {
  echo "Assemble docker command line flags"
  local docker_opts="-p /var/run/docker.pid --iptables=false --ip-masq=false"
  if [[ "${TEST_CLUSTER:-}" == "true" ]]; then
    docker_opts+=" --log-level=debug"
  else
    docker_opts+=" --log-level=warn"
  fi
  local use_net_plugin="true"
  if [[ "${NETWORK_PROVIDER:-}" == "kubenet" || "${NETWORK_PROVIDER:-}" == "cni" ]]; then
    # set docker0 cidr to private ip address range to avoid conflict with cbr0 cidr range
    docker_opts+=" --bip=169.254.123.1/24"
  else
    use_net_plugin="false"
    docker_opts+=" --bridge=cbr0"
  fi

  # Decide whether to enable a docker registry mirror. This is taken from
  # the "kube-env" metadata value.
  if [[ -n "${DOCKER_REGISTRY_MIRROR_URL:-}" ]]; then
    echo "Enable docker registry mirror at: ${DOCKER_REGISTRY_MIRROR_URL}"
    docker_opts+=" --registry-mirror=${DOCKER_REGISTRY_MIRROR_URL}"
  fi

  # Configure docker logging
  docker_opts+=" --log-driver=${DOCKER_LOG_DRIVER:-json-file}"
  docker_opts+=" --log-opt=max-size=${DOCKER_LOG_MAX_SIZE:-10m}"
  docker_opts+=" --log-opt=max-file=${DOCKER_LOG_MAX_FILE:-5}"

  # Disable live-restore if the environment variable is set.

  if [[ "${DISABLE_DOCKER_LIVE_RESTORE:-false}" == "true" ]]; then
    docker_opts+=" --live-restore=false"
  fi

  echo "DOCKER_OPTS=\"${docker_opts} ${EXTRA_DOCKER_OPTS:-}\"" > /etc/default/docker

  if [[ "${use_net_plugin}" == "true" ]]; then
    # If using a network plugin, extend the docker configuration to always remove
    # the network checkpoint to avoid corrupt checkpoints.
    # (https://github.com/docker/docker/issues/18283).
    echo "Extend the docker.service configuration to remove the network checkpiont"
    mkdir -p /etc/systemd/system/docker.service.d
    cat <<EOF >/etc/systemd/system/docker.service.d/01network.conf
[Service]
ExecStartPre=/bin/sh -x -c "rm -rf /var/lib/docker/network"
EOF
  fi

  # Ensure TasksMax is sufficient for docker.
  # (https://github.com/kubernetes/kubernetes/issues/51977)
  echo "Extend the docker.service configuration to set a higher pids limit"
  mkdir -p /etc/systemd/system/docker.service.d
  cat <<EOF >/etc/systemd/system/docker.service.d/02tasksmax.conf
[Service]
TasksMax=infinity
EOF

    systemctl daemon-reload
    echo "Docker command line is updated. Restart docker to pick it up"
    systemctl restart docker
}

# This function assembles the kubelet systemd service file and starts it
# using systemctl.
function start-kubelet {
  echo "Start kubelet"

  local -r kubelet_cert_dir="/var/lib/kubelet/pki/"
  mkdir -p "${kubelet_cert_dir}"

  local kubelet_bin="${KUBE_HOME}/bin/kubelet"
  local -r version="$("${kubelet_bin}" --version=true | cut -f2 -d " ")"
  local -r builtin_kubelet="/usr/bin/kubelet"
  if [[ "${TEST_CLUSTER:-}" == "true" ]]; then
    # Determine which binary to use on test clusters. We use the built-in
    # version only if the downloaded version is the same as the built-in
    # version. This allows GCI to run some of the e2e tests to qualify the
    # built-in kubelet.
    if [[ -x "${builtin_kubelet}" ]]; then
      local -r builtin_version="$("${builtin_kubelet}"  --version=true | cut -f2 -d " ")"
      if [[ "${builtin_version}" == "${version}" ]]; then
        kubelet_bin="${builtin_kubelet}"
      fi
    fi
  fi
  echo "Using kubelet binary at ${kubelet_bin}"
  local flags="${KUBELET_TEST_LOG_LEVEL:-"--v=2"} ${KUBELET_TEST_ARGS:-}"
  flags+=" --allow-privileged=true"
  flags+=" --cgroup-root=/"
  flags+=" --cloud-provider=gce"
  flags+=" --cluster-dns=${DNS_SERVER_IP}"
  flags+=" --cluster-domain=${DNS_DOMAIN}"
  flags+=" --pod-manifest-path=/etc/kubernetes/manifests"
  flags+=" --experimental-mounter-path=${CONTAINERIZED_MOUNTER_HOME}/mounter"
  flags+=" --experimental-check-node-capabilities-before-mount=true"
  flags+=" --cert-dir=${kubelet_cert_dir}"

  if [[ -n "${KUBELET_PORT:-}" ]]; then
    flags+=" --port=${KUBELET_PORT}"
  fi
  if [[ "${KUBERNETES_MASTER:-}" == "true" ]]; then
    flags+=" ${MASTER_KUBELET_TEST_ARGS:-}"
    flags+=" --enable-debugging-handlers=false"
    flags+=" --hairpin-mode=none"
    if [[ "${REGISTER_MASTER_KUBELET:-false}" == "true" ]]; then
      #TODO(mikedanese): allow static pods to start before creating a client
      #flags+=" --bootstrap-kubeconfig=/var/lib/kubelet/bootstrap-kubeconfig"
      #flags+=" --kubeconfig=/var/lib/kubelet/kubeconfig"
      flags+=" --kubeconfig=/var/lib/kubelet/bootstrap-kubeconfig"
      flags+=" --register-schedulable=false"
    else
      # Standalone mode (not widely used?)
      flags+=" --pod-cidr=${MASTER_IP_RANGE}"
    fi
  else # For nodes
    flags+=" ${NODE_KUBELET_TEST_ARGS:-}"
    flags+=" --enable-debugging-handlers=true"
    flags+=" --bootstrap-kubeconfig=/var/lib/kubelet/bootstrap-kubeconfig"
    flags+=" --kubeconfig=/var/lib/kubelet/kubeconfig"
    if [[ "${HAIRPIN_MODE:-}" == "promiscuous-bridge" ]] || \
       [[ "${HAIRPIN_MODE:-}" == "hairpin-veth" ]] || \
       [[ "${HAIRPIN_MODE:-}" == "none" ]]; then
      flags+=" --hairpin-mode=${HAIRPIN_MODE}"
    fi
    flags+=" --anonymous-auth=false --authorization-mode=Webhook --client-ca-file=${CA_CERT_BUNDLE_PATH}"
  fi
  # Network plugin
  if [[ -n "${NETWORK_PROVIDER:-}" || -n "${NETWORK_POLICY_PROVIDER:-}" ]]; then
    flags+=" --cni-bin-dir=/home/kubernetes/bin"
    if [[ "${NETWORK_POLICY_PROVIDER:-}" == "calico" ]]; then
      # Calico uses CNI always.
      # Keep KUBERNETES_PRIVATE_MASTER for backward compatibility.
      # Note that network policy won't work for master node.
      if [[ "${KUBERNETES_PRIVATE_MASTER:-}" == "true" || "${KUBERNETES_MASTER:-}" == "true" ]]; then
        flags+=" --network-plugin=${NETWORK_PROVIDER}"
      else
        flags+=" --network-plugin=cni"
      fi
    else
      # Otherwise use the configured value.
      flags+=" --network-plugin=${NETWORK_PROVIDER}"
    fi
  fi
  if [[ -n "${NON_MASQUERADE_CIDR:-}" ]]; then
    flags+=" --non-masquerade-cidr=${NON_MASQUERADE_CIDR}"
  fi
  # FlexVolume plugin
  if [[ -n "${VOLUME_PLUGIN_DIR:-}" ]]; then
    flags+=" --volume-plugin-dir=${VOLUME_PLUGIN_DIR}"
  fi
  if [[ "${ENABLE_MANIFEST_URL:-}" == "true" ]]; then
    flags+=" --manifest-url=${MANIFEST_URL}"
    flags+=" --manifest-url-header=${MANIFEST_URL_HEADER}"
  fi
  if [[ -n "${ENABLE_CUSTOM_METRICS:-}" ]]; then
    flags+=" --enable-custom-metrics=${ENABLE_CUSTOM_METRICS}"
  fi
  local node_labels=""
  if [[ "${KUBE_PROXY_DAEMONSET:-}" == "true" && "${KUBERNETES_MASTER:-}" != "true" ]]; then
    # Add kube-proxy daemonset label to node to avoid situation during cluster
    # upgrade/downgrade when there are two instances of kube-proxy running on a node.
    node_labels="beta.kubernetes.io/kube-proxy-ds-ready=true"
  fi
  if [[ -n "${NODE_LABELS:-}" ]]; then
    node_labels="${node_labels:+${node_labels},}${NODE_LABELS}"
  fi
  if [[ -n "${NON_MASTER_NODE_LABELS:-}" && "${KUBERNETES_MASTER:-}" != "true" ]]; then
    node_labels="${node_labels:+${node_labels},}${NON_MASTER_NODE_LABELS}"
  fi
  if [[ -n "${node_labels:-}" ]]; then
    flags+=" --node-labels=${node_labels}"
  fi
  if [[ -n "${NODE_TAINTS:-}" ]]; then
    flags+=" --register-with-taints=${NODE_TAINTS}"
  fi
  if [[ -n "${EVICTION_HARD:-}" ]]; then
    flags+=" --eviction-hard=${EVICTION_HARD}"
  fi
  if [[ -n "${FEATURE_GATES:-}" ]]; then
    flags+=" --feature-gates=${FEATURE_GATES}"
  fi
  if [[ -n "${ROTATE_CERTIFICATES:-}" ]]; then
    flags+=" --rotate-certificates=true"
  fi
  if [[ -n "${CONTAINER_RUNTIME:-}" ]]; then
    flags+=" --container-runtime=${CONTAINER_RUNTIME}"
  fi
  if [[ -n "${CONTAINER_RUNTIME_ENDPOINT:-}" ]]; then
    flags+=" --container-runtime-endpoint=${CONTAINER_RUNTIME_ENDPOINT}"
  fi


  local -r kubelet_env_file="/etc/default/kubelet"
  echo "KUBELET_OPTS=\"${flags}\"" > "${kubelet_env_file}"

  # Write the systemd service file for kubelet.
  cat <<EOF >/etc/systemd/system/kubelet.service
[Unit]
Description=Kubernetes kubelet
Requires=network-online.target
After=network-online.target

[Service]
Restart=always
RestartSec=10
EnvironmentFile=${kubelet_env_file}
ExecStart=${kubelet_bin} \$KUBELET_OPTS

[Install]
WantedBy=multi-user.target
EOF

  systemctl start kubelet.service
}

# This function assembles the node problem detector systemd service file and
# starts it using systemctl.
function start-node-problem-detector {
  echo "Start node problem detector"
  local -r npd_bin="${KUBE_HOME}/bin/node-problem-detector"
  local -r km_config="${KUBE_HOME}/node-problem-detector/config/kernel-monitor.json"
  # TODO(random-liu): Handle this for alternative container runtime.
  local -r dm_config="${KUBE_HOME}/node-problem-detector/config/docker-monitor.json"
  echo "Using node problem detector binary at ${npd_bin}"
  local flags="${NPD_TEST_LOG_LEVEL:-"--v=2"} ${NPD_TEST_ARGS:-}"
  flags+=" --logtostderr"
  flags+=" --system-log-monitors=${km_config},${dm_config}"
  flags+=" --apiserver-override=https://${KUBERNETES_MASTER_NAME}?inClusterConfig=false&auth=/var/lib/node-problem-detector/kubeconfig"
  local -r npd_port=${NODE_PROBLEM_DETECTOR_PORT:-20256}
  flags+=" --port=${npd_port}"

  # Write the systemd service file for node problem detector.
  cat <<EOF >/etc/systemd/system/node-problem-detector.service
[Unit]
Description=Kubernetes node problem detector
Requires=network-online.target
After=network-online.target

[Service]
Restart=always
RestartSec=10
ExecStart=${npd_bin} ${flags}

[Install]
WantedBy=multi-user.target
EOF

  systemctl start node-problem-detector.service
}

# Create the log file and set its properties.
#
# $1 is the file to create.
function prepare-log-file {
  touch $1
  chmod 644 $1
  chown root:root $1
}

# Prepares parameters for kube-proxy manifest.
# $1 source path of kube-proxy manifest.
function prepare-kube-proxy-manifest-variables {
  local -r src_file=$1;

  remove-salt-config-comments "${src_file}"

  local -r kubeconfig="--kubeconfig=/var/lib/kube-proxy/kubeconfig"
  local kube_docker_registry="gcr.io/google_containers"
  if [[ -n "${KUBE_DOCKER_REGISTRY:-}" ]]; then
    kube_docker_registry=${KUBE_DOCKER_REGISTRY}
  fi
  local -r kube_proxy_docker_tag=$(cat /home/kubernetes/kube-docker-files/kube-proxy.docker_tag)
  local api_servers="--master=https://${KUBERNETES_MASTER_NAME}"
  local params="${KUBEPROXY_TEST_LOG_LEVEL:-"--v=2"}"
  if [[ -n "${FEATURE_GATES:-}" ]]; then
    params+=" --feature-gates=${FEATURE_GATES}"
  fi
  params+=" --iptables-sync-period=1m --iptables-min-sync-period=10s --ipvs-sync-period=1m --ipvs-min-sync-period=10s"
  if [[ -n "${KUBEPROXY_TEST_ARGS:-}" ]]; then
    params+=" ${KUBEPROXY_TEST_ARGS}"
  fi
  local container_env=""
  local kube_cache_mutation_detector_env_name=""
  local kube_cache_mutation_detector_env_value=""
  if [[ -n "${ENABLE_CACHE_MUTATION_DETECTOR:-}" ]]; then
    container_env="env:"
    kube_cache_mutation_detector_env_name="- name: KUBE_CACHE_MUTATION_DETECTOR"
    kube_cache_mutation_detector_env_value="value: \"${ENABLE_CACHE_MUTATION_DETECTOR}\""
  fi
  local pod_priority=""
  if [[ "${ENABLE_POD_PRIORITY:-}" == "true" ]]; then
    pod_priority="priorityClassName: system-node-critical"
  fi
  sed -i -e "s@{{kubeconfig}}@${kubeconfig}@g" ${src_file}
  sed -i -e "s@{{pillar\['kube_docker_registry'\]}}@${kube_docker_registry}@g" ${src_file}
  sed -i -e "s@{{pillar\['kube-proxy_docker_tag'\]}}@${kube_proxy_docker_tag}@g" ${src_file}
  sed -i -e "s@{{params}}@${params}@g" ${src_file}
  sed -i -e "s@{{container_env}}@${container_env}@g" ${src_file}
  sed -i -e "s@{{kube_cache_mutation_detector_env_name}}@${kube_cache_mutation_detector_env_name}@g" ${src_file}
  sed -i -e "s@{{kube_cache_mutation_detector_env_value}}@${kube_cache_mutation_detector_env_value}@g" ${src_file}
  sed -i -e "s@{{pod_priority}}@${pod_priority}@g" ${src_file}
  sed -i -e "s@{{ cpurequest }}@100m@g" ${src_file}
  sed -i -e "s@{{api_servers_with_port}}@${api_servers}@g" ${src_file}
  sed -i -e "s@{{kubernetes_service_host_env_value}}@${KUBERNETES_MASTER_NAME}@g" ${src_file}
  if [[ -n "${CLUSTER_IP_RANGE:-}" ]]; then
    sed -i -e "s@{{cluster_cidr}}@--cluster-cidr=${CLUSTER_IP_RANGE}@g" ${src_file}
  fi
}

# Starts kube-proxy static pod.
function start-kube-proxy {
  echo "Start kube-proxy static pod"
  prepare-log-file /var/log/kube-proxy.log
  local -r src_file="${KUBE_HOME}/kube-manifests/kubernetes/kube-proxy.manifest"
  prepare-kube-proxy-manifest-variables "${src_file}"

  cp "${src_file}" /etc/kubernetes/manifests
}

# Replaces the variables in the etcd manifest file with the real values, and then
# copy the file to the manifest dir
# $1: value for variable 'suffix'
# $2: value for variable 'port'
# $3: value for variable 'server_port'
# $4: value for variable 'cpulimit'
# $5: pod name, which should be either etcd or etcd-events
function prepare-etcd-manifest {
  local host_name=${ETCD_HOSTNAME:-$(hostname -s)}
  local etcd_cluster=""
  local cluster_state="new"
  local etcd_protocol="http"
  local etcd_creds=""

  if [[ -n "${INITIAL_ETCD_CLUSTER_STATE:-}" ]]; then
    cluster_state="${INITIAL_ETCD_CLUSTER_STATE}"
  fi
  if [[ -n "${ETCD_CA_KEY:-}" && -n "${ETCD_CA_CERT:-}" && -n "${ETCD_PEER_KEY:-}" && -n "${ETCD_PEER_CERT:-}" ]]; then
    etcd_creds=" --peer-trusted-ca-file /etc/srv/kubernetes/etcd-ca.crt --peer-cert-file /etc/srv/kubernetes/etcd-peer.crt --peer-key-file /etc/srv/kubernetes/etcd-peer.key -peer-client-cert-auth "
    etcd_protocol="https"
  fi

  for host in $(echo "${INITIAL_ETCD_CLUSTER:-${host_name}}" | tr "," "\n"); do
    etcd_host="etcd-${host}=${etcd_protocol}://${host}:$3"
    if [[ -n "${etcd_cluster}" ]]; then
      etcd_cluster+=","
    fi
    etcd_cluster+="${etcd_host}"
  done

  local -r temp_file="/tmp/$5"
  cp "${KUBE_HOME}/kube-manifests/kubernetes/gci-trusty/etcd.manifest" "${temp_file}"
  remove-salt-config-comments "${temp_file}"
  sed -i -e "s@{{ *suffix *}}@$1@g" "${temp_file}"
  sed -i -e "s@{{ *port *}}@$2@g" "${temp_file}"
  sed -i -e "s@{{ *server_port *}}@$3@g" "${temp_file}"
  sed -i -e "s@{{ *cpulimit *}}@\"$4\"@g" "${temp_file}"
  sed -i -e "s@{{ *hostname *}}@$host_name@g" "${temp_file}"
  sed -i -e "s@{{ *srv_kube_path *}}@/etc/srv/kubernetes@g" "${temp_file}"
  sed -i -e "s@{{ *etcd_cluster *}}@$etcd_cluster@g" "${temp_file}"
  sed -i -e "s@{{ *liveness_probe_initial_delay *}}@${ETCD_LIVENESS_PROBE_INITIAL_DELAY_SEC:-15}@g" "${temp_file}"
  # Get default storage backend from manifest file.
  local -r default_storage_backend=$(cat "${temp_file}" | \
    grep -o "{{ *pillar\.get('storage_backend', '\(.*\)') *}}" | \
    sed -e "s@{{ *pillar\.get('storage_backend', '\(.*\)') *}}@\1@g")
  if [[ -n "${STORAGE_BACKEND:-}" ]]; then
    sed -i -e "s@{{ *pillar\.get('storage_backend', '\(.*\)') *}}@${STORAGE_BACKEND}@g" "${temp_file}"
  else
    sed -i -e "s@{{ *pillar\.get('storage_backend', '\(.*\)') *}}@\1@g" "${temp_file}"
  fi
  if [[ "${STORAGE_BACKEND:-${default_storage_backend}}" == "etcd3" ]]; then
    sed -i -e "s@{{ *quota_bytes *}}@--quota-backend-bytes=${ETCD_QUOTA_BACKEND_BYTES:-4294967296}@g" "${temp_file}"
  else
    sed -i -e "s@{{ *quota_bytes *}}@@g" "${temp_file}"
  fi
  sed -i -e "s@{{ *cluster_state *}}@$cluster_state@g" "${temp_file}"
  if [[ -n "${ETCD_IMAGE:-}" ]]; then
    sed -i -e "s@{{ *pillar\.get('etcd_docker_tag', '\(.*\)') *}}@${ETCD_IMAGE}@g" "${temp_file}"
  else
    sed -i -e "s@{{ *pillar\.get('etcd_docker_tag', '\(.*\)') *}}@\1@g" "${temp_file}"
  fi
  if [[ -n "${ETCD_DOCKER_REPOSITORY:-}" ]]; then
    sed -i -e "s@{{ *pillar\.get('etcd_docker_repository', '\(.*\)') *}}@${ETCD_DOCKER_REPOSITORY}@g" "${temp_file}"
  else
    sed -i -e "s@{{ *pillar\.get('etcd_docker_repository', '\(.*\)') *}}@\1@g" "${temp_file}"
  fi
  sed -i -e "s@{{ *etcd_protocol *}}@$etcd_protocol@g" "${temp_file}"
  sed -i -e "s@{{ *etcd_creds *}}@$etcd_creds@g" "${temp_file}"
  if [[ -n "${ETCD_VERSION:-}" ]]; then
    sed -i -e "s@{{ *pillar\.get('etcd_version', '\(.*\)') *}}@${ETCD_VERSION}@g" "${temp_file}"
  else
    sed -i -e "s@{{ *pillar\.get('etcd_version', '\(.*\)') *}}@\1@g" "${temp_file}"
  fi
  # Replace the volume host path.
  sed -i -e "s@/mnt/master-pd/var/etcd@/mnt/disks/master-pd/var/etcd@g" "${temp_file}"
  mv "${temp_file}" /etc/kubernetes/manifests
}

function start-etcd-empty-dir-cleanup-pod {
  cp "${KUBE_HOME}/kube-manifests/kubernetes/gci-trusty/etcd-empty-dir-cleanup/etcd-empty-dir-cleanup.yaml" "/etc/kubernetes/manifests"
}

# Starts etcd server pod (and etcd-events pod if needed).
# More specifically, it prepares dirs and files, sets the variable value
# in the manifests, and copies them to /etc/kubernetes/manifests.
function start-etcd-servers {
  echo "Start etcd pods"
  if [[ -d /etc/etcd ]]; then
    rm -rf /etc/etcd
  fi
  if [[ -e /etc/default/etcd ]]; then
    rm -f /etc/default/etcd
  fi
  if [[ -e /etc/systemd/system/etcd.service ]]; then
    rm -f /etc/systemd/system/etcd.service
  fi
  if [[ -e /etc/init.d/etcd ]]; then
    rm -f /etc/init.d/etcd
  fi
  prepare-log-file /var/log/etcd.log
  prepare-etcd-manifest "" "2379" "2380" "200m" "etcd.manifest"

  prepare-log-file /var/log/etcd-events.log
  prepare-etcd-manifest "-events" "4002" "2381" "100m" "etcd-events.manifest"
}

# Calculates the following variables based on env variables, which will be used
# by the manifests of several kube-master components.
#   CLOUD_CONFIG_OPT
#   CLOUD_CONFIG_VOLUME
#   CLOUD_CONFIG_MOUNT
#   DOCKER_REGISTRY
#   FLEXVOLUME_HOSTPATH_MOUNT
#   FLEXVOLUME_HOSTPATH_VOLUME
function compute-master-manifest-variables {
  CLOUD_CONFIG_OPT=""
  CLOUD_CONFIG_VOLUME=""
  CLOUD_CONFIG_MOUNT=""
  if [[ -f /etc/gce.conf ]]; then
    CLOUD_CONFIG_OPT="--cloud-config=/etc/gce.conf"
    CLOUD_CONFIG_VOLUME="{\"name\": \"cloudconfigmount\",\"hostPath\": {\"path\": \"/etc/gce.conf\", \"type\": \"FileOrCreate\"}},"
    CLOUD_CONFIG_MOUNT="{\"name\": \"cloudconfigmount\",\"mountPath\": \"/etc/gce.conf\", \"readOnly\": true},"
  fi
  DOCKER_REGISTRY="gcr.io/google_containers"
  if [[ -n "${KUBE_DOCKER_REGISTRY:-}" ]]; then
    DOCKER_REGISTRY="${KUBE_DOCKER_REGISTRY}"
  fi

  FLEXVOLUME_HOSTPATH_MOUNT=""
  FLEXVOLUME_HOSTPATH_VOLUME=""
  if [[ -n "${VOLUME_PLUGIN_DIR:-}" ]]; then
    FLEXVOLUME_HOSTPATH_MOUNT="{ \"name\": \"flexvolumedir\", \"mountPath\": \"${VOLUME_PLUGIN_DIR}\", \"readOnly\": true},"
    FLEXVOLUME_HOSTPATH_VOLUME="{ \"name\": \"flexvolumedir\", \"hostPath\": {\"path\": \"${VOLUME_PLUGIN_DIR}\"}},"
  fi
}

# A helper function that bind mounts kubelet dirs for running mount in a chroot
function prepare-mounter-rootfs {
  echo "Prepare containerized mounter"
  mount --bind "${CONTAINERIZED_MOUNTER_HOME}" "${CONTAINERIZED_MOUNTER_HOME}"
  mount -o remount,exec "${CONTAINERIZED_MOUNTER_HOME}"
  CONTAINERIZED_MOUNTER_ROOTFS="${CONTAINERIZED_MOUNTER_HOME}/rootfs"
  mount --rbind /var/lib/kubelet/ "${CONTAINERIZED_MOUNTER_ROOTFS}/var/lib/kubelet"
  mount --make-rshared "${CONTAINERIZED_MOUNTER_ROOTFS}/var/lib/kubelet"
  mount --bind -o ro /proc "${CONTAINERIZED_MOUNTER_ROOTFS}/proc"
  mount --bind -o ro /dev "${CONTAINERIZED_MOUNTER_ROOTFS}/dev"
  cp /etc/resolv.conf "${CONTAINERIZED_MOUNTER_ROOTFS}/etc/"
}

# A helper function for removing salt configuration and comments from a file.
# This is mainly for preparing a manifest file.
#
# $1: Full path of the file to manipulate
function remove-salt-config-comments {
  # Remove salt configuration.
  sed -i "/^[ |\t]*{[#|%]/d" $1
  # Remove comments.
  sed -i "/^[ |\t]*#/d" $1
}

# Starts kubernetes apiserver.
# It prepares the log file, loads the docker image, calculates variables, sets them
# in the manifest file, and then copies the manifest file to /etc/kubernetes/manifests.
#
# Assumed vars (which are calculated in function compute-master-manifest-variables)
#   CLOUD_CONFIG_OPT
#   CLOUD_CONFIG_VOLUME
#   CLOUD_CONFIG_MOUNT
#   DOCKER_REGISTRY
function start-kube-apiserver {
  echo "Start kubernetes api-server"
  prepare-log-file /var/log/kube-apiserver.log
  prepare-log-file /var/log/kube-apiserver-audit.log

  # Calculate variables and assemble the command line.
  local params="${API_SERVER_TEST_LOG_LEVEL:-"--v=2"} ${APISERVER_TEST_ARGS:-} ${CLOUD_CONFIG_OPT}"
  params+=" --address=127.0.0.1"
  params+=" --allow-privileged=true"
  params+=" --cloud-provider=gce"
  params+=" --client-ca-file=${CA_CERT_BUNDLE_PATH}"
  params+=" --etcd-servers=http://127.0.0.1:2379"
  params+=" --etcd-servers-overrides=/events#http://127.0.0.1:4002"
  params+=" --secure-port=443"
  params+=" --tls-cert-file=${APISERVER_SERVER_CERT_PATH}"
  params+=" --tls-private-key-file=${APISERVER_SERVER_KEY_PATH}"
  params+=" --kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname"
  if [[ -s "${REQUESTHEADER_CA_CERT_PATH:-}" ]]; then
    params+=" --requestheader-client-ca-file=${REQUESTHEADER_CA_CERT_PATH}"
    params+=" --requestheader-allowed-names=aggregator"
    params+=" --requestheader-extra-headers-prefix=X-Remote-Extra-"
    params+=" --requestheader-group-headers=X-Remote-Group"
    params+=" --requestheader-username-headers=X-Remote-User"
    params+=" --proxy-client-cert-file=${PROXY_CLIENT_CERT_PATH}"
    params+=" --proxy-client-key-file=${PROXY_CLIENT_KEY_PATH}"
  fi
  params+=" --enable-aggregator-routing=true"
  if [[ -e "${APISERVER_CLIENT_CERT_PATH}" ]] && [[ -e "${APISERVER_CLIENT_KEY_PATH}" ]]; then
    params+=" --kubelet-client-certificate=${APISERVER_CLIENT_CERT_PATH}"
    params+=" --kubelet-client-key=${APISERVER_CLIENT_KEY_PATH}"
  fi
  if [[ -n "${SERVICEACCOUNT_CERT_PATH:-}" ]]; then
    params+=" --service-account-key-file=${SERVICEACCOUNT_CERT_PATH}"
  fi
  params+=" --token-auth-file=/etc/srv/kubernetes/known_tokens.csv"
  if [[ -n "${KUBE_PASSWORD:-}" && -n "${KUBE_USER:-}" ]]; then
    params+=" --basic-auth-file=/etc/srv/kubernetes/basic_auth.csv"
  fi
  if [[ -n "${STORAGE_BACKEND:-}" ]]; then
    params+=" --storage-backend=${STORAGE_BACKEND}"
  fi
  if [[ -n "${STORAGE_MEDIA_TYPE:-}" ]]; then
    params+=" --storage-media-type=${STORAGE_MEDIA_TYPE}"
  fi
  if [[ -n "${ETCD_COMPACTION_INTERVAL_SEC:-}" ]]; then
    params+=" --etcd-compaction-interval=${ETCD_COMPACTION_INTERVAL_SEC}s"
  fi
  if [[ -n "${KUBE_APISERVER_REQUEST_TIMEOUT_SEC:-}" ]]; then
    params+=" --request-timeout=${KUBE_APISERVER_REQUEST_TIMEOUT_SEC}s"
  fi
  if [[ -n "${ENABLE_GARBAGE_COLLECTOR:-}" ]]; then
    params+=" --enable-garbage-collector=${ENABLE_GARBAGE_COLLECTOR}"
  fi
  if [[ -n "${NUM_NODES:-}" ]]; then
    # If the cluster is large, increase max-requests-inflight limit in apiserver.
    if [[ "${NUM_NODES}" -ge 1000 ]]; then
      params+=" --max-requests-inflight=1500 --max-mutating-requests-inflight=500"
    fi
    # Set amount of memory available for apiserver based on number of nodes.
    # TODO: Once we start setting proper requests and limits for apiserver
    # we should reuse the same logic here instead of current heuristic.
    params+=" --target-ram-mb=$((${NUM_NODES} * 60))"
  fi
  if [[ -n "${SERVICE_CLUSTER_IP_RANGE:-}" ]]; then
    params+=" --service-cluster-ip-range=${SERVICE_CLUSTER_IP_RANGE}"
  fi
  if [[ -n "${ETCD_QUORUM_READ:-}" ]]; then
    params+=" --etcd-quorum-read=${ETCD_QUORUM_READ}"
  fi

  local audit_policy_config_mount=""
  local audit_policy_config_volume=""
  local audit_webhook_config_mount=""
  local audit_webhook_config_volume=""
  if [[ "${ENABLE_APISERVER_BASIC_AUDIT:-}" == "true" ]]; then
    # We currently only support enabling with a fixed path and with built-in log
    # rotation "disabled" (large value) so it behaves like kube-apiserver.log.
    # External log rotation should be set up the same as for kube-apiserver.log.
    params+=" --audit-log-path=/var/log/kube-apiserver-audit.log"
    params+=" --audit-log-maxage=0"
    params+=" --audit-log-maxbackup=0"
    # Lumberjack doesn't offer any way to disable size-based rotation. It also
    # has an in-memory counter that doesn't notice if you truncate the file.
    # 2000000000 (in MiB) is a large number that fits in 31 bits. If the log
    # grows at 10MiB/s (~30K QPS), it will rotate after ~6 years if apiserver
    # never restarts. Please manually restart apiserver before this time.
    params+=" --audit-log-maxsize=2000000000"
    # Disable AdvancedAuditing enabled by default
    if [[ -z "${FEATURE_GATES:-}" ]]; then
      FEATURE_GATES="AdvancedAuditing=false"
    else
      FEATURE_GATES="${FEATURE_GATES},AdvancedAuditing=false"
    fi
  elif [[ "${ENABLE_APISERVER_ADVANCED_AUDIT:-}" == "true" ]]; then
    local -r audit_policy_file="/etc/audit_policy.config"
    params+=" --audit-policy-file=${audit_policy_file}"
    # Create the audit policy file, and mount it into the apiserver pod.
    create-master-audit-policy "${audit_policy_file}" "${ADVANCED_AUDIT_POLICY:-}"
    audit_policy_config_mount="{\"name\": \"auditpolicyconfigmount\",\"mountPath\": \"${audit_policy_file}\", \"readOnly\": true},"
    audit_policy_config_volume="{\"name\": \"auditpolicyconfigmount\",\"hostPath\": {\"path\": \"${audit_policy_file}\", \"type\": \"FileOrCreate\"}},"

    if [[ "${ADVANCED_AUDIT_BACKEND:-log}" == *"log"* ]]; then
      # The advanced audit log backend config matches the basic audit log config.
      params+=" --audit-log-path=/var/log/kube-apiserver-audit.log"
      params+=" --audit-log-maxage=0"
      params+=" --audit-log-maxbackup=0"
      # Lumberjack doesn't offer any way to disable size-based rotation. It also
      # has an in-memory counter that doesn't notice if you truncate the file.
      # 2000000000 (in MiB) is a large number that fits in 31 bits. If the log
      # grows at 10MiB/s (~30K QPS), it will rotate after ~6 years if apiserver
      # never restarts. Please manually restart apiserver before this time.
      params+=" --audit-log-maxsize=2000000000"
    fi
    if [[ "${ADVANCED_AUDIT_BACKEND:-}" == *"webhook"* ]]; then
      params+=" --audit-webhook-mode=batch"

      # Create the audit webhook config file, and mount it into the apiserver pod.
      local -r audit_webhook_config_file="/etc/audit_webhook.config"
      params+=" --audit-webhook-config-file=${audit_webhook_config_file}"
      if [[ -n "${ADVANCED_AUDIT_WEBHOOK_BUFFER_SIZE:-}" ]]; then
        params+=" --audit-webhook-batch-buffer-size=${ADVANCED_AUDIT_WEBHOOK_BUFFER_SIZE}"
      fi
      if [[ -n "${ADVANCED_AUDIT_WEBHOOK_MAX_BATCH_SIZE:-}" ]]; then
        params+=" --audit-webhook-batch-max-size=${ADVANCED_AUDIT_WEBHOOK_MAX_BATCH_SIZE}"
      fi
      if [[ -n "${ADVANCED_AUDIT_WEBHOOK_MAX_BATCH_WAIT:-}" ]]; then
        params+=" --audit-webhook-batch-max-wait=${ADVANCED_AUDIT_WEBHOOK_MAX_BATCH_WAIT}"
      fi
      if [[ -n "${ADVANCED_AUDIT_WEBHOOK_THROTTLE_QPS:-}" ]]; then
        params+=" --audit-webhook-batch-throttle-qps=${ADVANCED_AUDIT_WEBHOOK_THROTTLE_QPS}"
      fi
      if [[ -n "${ADVANCED_AUDIT_WEBHOOK_THROTTLE_BURST:-}" ]]; then
        params+=" --audit-webhook-batch-throttle-burst=${ADVANCED_AUDIT_WEBHOOK_THROTTLE_BURST}"
      fi
      if [[ -n "${ADVANCED_AUDIT_WEBHOOK_INITIAL_BACKOFF:-}" ]]; then
        params+=" --audit-webhook-batch-initial-backoff=${ADVANCED_AUDIT_WEBHOOK_INITIAL_BACKOFF}"
      fi
      create-master-audit-webhook-config "${audit_webhook_config_file}"
      audit_webhook_config_mount="{\"name\": \"auditwebhookconfigmount\",\"mountPath\": \"${audit_webhook_config_file}\", \"readOnly\": true},"
      audit_webhook_config_volume="{\"name\": \"auditwebhookconfigmount\",\"hostPath\": {\"path\": \"${audit_webhook_config_file}\", \"type\": \"FileOrCreate\"}},"
    fi
  fi

  if [[ "${ENABLE_APISERVER_LOGS_HANDLER:-}" == "false" ]]; then
    params+=" --enable-logs-handler=false"
  fi

  local admission_controller_config_mount=""
  local admission_controller_config_volume=""
  local image_policy_webhook_config_mount=""
  local image_policy_webhook_config_volume=""
  if [[ -n "${ADMISSION_CONTROL:-}" ]]; then
    params+=" --admission-control=${ADMISSION_CONTROL}"
    if [[ ${ADMISSION_CONTROL} == *"ImagePolicyWebhook"* ]]; then
      params+=" --admission-control-config-file=/etc/admission_controller.config"
      # Mount the file to configure admission controllers if ImagePolicyWebhook is set.
      admission_controller_config_mount="{\"name\": \"admissioncontrollerconfigmount\",\"mountPath\": \"/etc/admission_controller.config\", \"readOnly\": false},"
      admission_controller_config_volume="{\"name\": \"admissioncontrollerconfigmount\",\"hostPath\": {\"path\": \"/etc/admission_controller.config\", \"type\": \"FileOrCreate\"}},"
      # Mount the file to configure the ImagePolicyWebhook's webhook.
      image_policy_webhook_config_mount="{\"name\": \"imagepolicywebhookconfigmount\",\"mountPath\": \"/etc/gcp_image_review.config\", \"readOnly\": false},"
      image_policy_webhook_config_volume="{\"name\": \"imagepolicywebhookconfigmount\",\"hostPath\": {\"path\": \"/etc/gcp_image_review.config\", \"type\": \"FileOrCreate\"}},"
    fi
  fi

  if [[ -n "${KUBE_APISERVER_REQUEST_TIMEOUT:-}" ]]; then
    params+=" --min-request-timeout=${KUBE_APISERVER_REQUEST_TIMEOUT}"
  fi
  if [[ -n "${RUNTIME_CONFIG:-}" ]]; then
    params+=" --runtime-config=${RUNTIME_CONFIG}"
  fi
  if [[ -n "${FEATURE_GATES:-}" ]]; then
    params+=" --feature-gates=${FEATURE_GATES}"
  fi
  if [[ -n "${PROJECT_ID:-}" && -n "${TOKEN_URL:-}" && -n "${TOKEN_BODY:-}" && -n "${NODE_NETWORK:-}" ]]; then
    local -r vm_external_ip=$(curl --retry 5 --retry-delay 3 ${CURL_RETRY_CONNREFUSED} --fail --silent -H 'Metadata-Flavor: Google' "http://metadata/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip")
    if [[ -n "${PROXY_SSH_USER:-}" ]]; then
      params+=" --advertise-address=${vm_external_ip}"      
      params+=" --ssh-user=${PROXY_SSH_USER}"
      params+=" --ssh-keyfile=/etc/srv/sshproxy/.sshkeyfile"
    fi
  elif [ -n "${MASTER_ADVERTISE_ADDRESS:-}" ]; then
    params="${params} --advertise-address=${MASTER_ADVERTISE_ADDRESS}"
  fi

  local webhook_authn_config_mount=""
  local webhook_authn_config_volume=""
  if [[ -n "${GCP_AUTHN_URL:-}" ]]; then
    params+=" --authentication-token-webhook-config-file=/etc/gcp_authn.config"
    webhook_authn_config_mount="{\"name\": \"webhookauthnconfigmount\",\"mountPath\": \"/etc/gcp_authn.config\", \"readOnly\": false},"
    webhook_authn_config_volume="{\"name\": \"webhookauthnconfigmount\",\"hostPath\": {\"path\": \"/etc/gcp_authn.config\", \"type\": \"FileOrCreate\"}},"
  fi


  local authorization_mode="Node,RBAC"
  local -r src_dir="${KUBE_HOME}/kube-manifests/kubernetes/gci-trusty"

  # Enable ABAC mode unless the user explicitly opts out with ENABLE_LEGACY_ABAC=false
  if [[ "${ENABLE_LEGACY_ABAC:-}" != "false" ]]; then
    echo "Warning: Enabling legacy ABAC policy. All service accounts will have superuser API access. Set ENABLE_LEGACY_ABAC=false to disable this."
    # Create the ABAC file if it doesn't exist yet, or if we have a KUBE_USER set (to ensure the right user is given permissions)
    if [[ -n "${KUBE_USER:-}" || ! -e /etc/srv/kubernetes/abac-authz-policy.jsonl ]]; then
      local -r abac_policy_json="${src_dir}/abac-authz-policy.jsonl"
      remove-salt-config-comments "${abac_policy_json}"
      if [[ -n "${KUBE_USER:-}" ]]; then
        sed -i -e "s/{{kube_user}}/${KUBE_USER}/g" "${abac_policy_json}"
      else
        sed -i -e "/{{kube_user}}/d" "${abac_policy_json}"
      fi
      cp "${abac_policy_json}" /etc/srv/kubernetes/
    fi

    params+=" --authorization-policy-file=/etc/srv/kubernetes/abac-authz-policy.jsonl"
    authorization_mode+=",ABAC"
  fi

  local webhook_config_mount=""
  local webhook_config_volume=""
  if [[ -n "${GCP_AUTHZ_URL:-}" ]]; then
    authorization_mode+=",Webhook"
    params+=" --authorization-webhook-config-file=/etc/gcp_authz.config"
    webhook_config_mount="{\"name\": \"webhookconfigmount\",\"mountPath\": \"/etc/gcp_authz.config\", \"readOnly\": false},"
    webhook_config_volume="{\"name\": \"webhookconfigmount\",\"hostPath\": {\"path\": \"/etc/gcp_authz.config\", \"type\": \"FileOrCreate\"}},"
  fi
  params+=" --authorization-mode=${authorization_mode}"

  local container_env=""
  if [[ -n "${ENABLE_CACHE_MUTATION_DETECTOR:-}" ]]; then
    container_env="\"name\": \"KUBE_CACHE_MUTATION_DETECTOR\", \"value\": \"${ENABLE_CACHE_MUTATION_DETECTOR}\""
  fi
  if [[ -n "${ENABLE_PATCH_CONVERSION_DETECTOR:-}" ]]; then
    if [[ -n "${container_env}" ]]; then
      container_env="${container_env}, "
    fi
    container_env="\"name\": \"KUBE_PATCH_CONVERSION_DETECTOR\", \"value\": \"${ENABLE_PATCH_CONVERSION_DETECTOR}\""
  fi
  if [[ -n "${container_env}" ]]; then
    container_env="\"env\":[{${container_env}}],"
  fi

  if [[ -n "${ENCRYPTION_PROVIDER_CONFIG:-}" ]]; then
    local encryption_provider_config_path="/etc/srv/kubernetes/encryption-provider-config.yml"
    echo "${ENCRYPTION_PROVIDER_CONFIG}" | base64 --decode > "${encryption_provider_config_path}"
    params+=" --experimental-encryption-provider-config=${encryption_provider_config_path}"
  fi

  src_file="${src_dir}/kube-apiserver.manifest"
  remove-salt-config-comments "${src_file}"
  # Evaluate variables.
  local -r kube_apiserver_docker_tag=$(cat /home/kubernetes/kube-docker-files/kube-apiserver.docker_tag)
  sed -i -e "s@{{params}}@${params}@g" "${src_file}"
  sed -i -e "s@{{container_env}}@${container_env}@g" ${src_file}
  sed -i -e "s@{{srv_kube_path}}@/etc/srv/kubernetes@g" "${src_file}"
  sed -i -e "s@{{srv_sshproxy_path}}@/etc/srv/sshproxy@g" "${src_file}"
  sed -i -e "s@{{cloud_config_mount}}@${CLOUD_CONFIG_MOUNT}@g" "${src_file}"
  sed -i -e "s@{{cloud_config_volume}}@${CLOUD_CONFIG_VOLUME}@g" "${src_file}"
  sed -i -e "s@{{pillar\['kube_docker_registry'\]}}@${DOCKER_REGISTRY}@g" "${src_file}"
  sed -i -e "s@{{pillar\['kube-apiserver_docker_tag'\]}}@${kube_apiserver_docker_tag}@g" "${src_file}"
  sed -i -e "s@{{pillar\['allow_privileged'\]}}@true@g" "${src_file}"
  sed -i -e "s@{{liveness_probe_initial_delay}}@${KUBE_APISERVER_LIVENESS_PROBE_INITIAL_DELAY_SEC:-15}@g" "${src_file}"
  sed -i -e "s@{{secure_port}}@443@g" "${src_file}"
  sed -i -e "s@{{secure_port}}@8080@g" "${src_file}"
  sed -i -e "s@{{additional_cloud_config_mount}}@@g" "${src_file}"
  sed -i -e "s@{{additional_cloud_config_volume}}@@g" "${src_file}"
  sed -i -e "s@{{webhook_authn_config_mount}}@${webhook_authn_config_mount}@g" "${src_file}"
  sed -i -e "s@{{webhook_authn_config_volume}}@${webhook_authn_config_volume}@g" "${src_file}"
  sed -i -e "s@{{webhook_config_mount}}@${webhook_config_mount}@g" "${src_file}"
  sed -i -e "s@{{webhook_config_volume}}@${webhook_config_volume}@g" "${src_file}"
  sed -i -e "s@{{audit_policy_config_mount}}@${audit_policy_config_mount}@g" "${src_file}"
  sed -i -e "s@{{audit_policy_config_volume}}@${audit_policy_config_volume}@g" "${src_file}"
  sed -i -e "s@{{audit_webhook_config_mount}}@${audit_webhook_config_mount}@g" "${src_file}"
  sed -i -e "s@{{audit_webhook_config_volume}}@${audit_webhook_config_volume}@g" "${src_file}"
  sed -i -e "s@{{admission_controller_config_mount}}@${admission_controller_config_mount}@g" "${src_file}"
  sed -i -e "s@{{admission_controller_config_volume}}@${admission_controller_config_volume}@g" "${src_file}"
  sed -i -e "s@{{image_policy_webhook_config_mount}}@${image_policy_webhook_config_mount}@g" "${src_file}"
  sed -i -e "s@{{image_policy_webhook_config_volume}}@${image_policy_webhook_config_volume}@g" "${src_file}"
  cp "${src_file}" /etc/kubernetes/manifests
}

# Starts kubernetes controller manager.
# It prepares the log file, loads the docker image, calculates variables, sets them
# in the manifest file, and then copies the manifest file to /etc/kubernetes/manifests.
#
# Assumed vars (which are calculated in function compute-master-manifest-variables)
#   CLOUD_CONFIG_OPT
#   CLOUD_CONFIG_VOLUME
#   CLOUD_CONFIG_MOUNT
#   DOCKER_REGISTRY
function start-kube-controller-manager {
  echo "Start kubernetes controller-manager"
  create-kubecontrollermanager-kubeconfig
  prepare-log-file /var/log/kube-controller-manager.log
  # Calculate variables and assemble the command line.
  local params="${CONTROLLER_MANAGER_TEST_LOG_LEVEL:-"--v=2"} ${CONTROLLER_MANAGER_TEST_ARGS:-} ${CLOUD_CONFIG_OPT}"
  params+=" --use-service-account-credentials"
  params+=" --cloud-provider=gce"
  params+=" --kubeconfig=/etc/srv/kubernetes/kube-controller-manager/kubeconfig"
  params+=" --root-ca-file=${CA_CERT_BUNDLE_PATH}"
  params+=" --service-account-private-key-file=${SERVICEACCOUNT_KEY_PATH}"
  if [[ -n "${ENABLE_GARBAGE_COLLECTOR:-}" ]]; then
    params+=" --enable-garbage-collector=${ENABLE_GARBAGE_COLLECTOR}"
  fi
  if [[ -n "${INSTANCE_PREFIX:-}" ]]; then
    params+=" --cluster-name=${INSTANCE_PREFIX}"
  fi
  if [[ -n "${CLUSTER_IP_RANGE:-}" ]]; then
    params+=" --cluster-cidr=${CLUSTER_IP_RANGE}"
  fi
  if [[ -n "${CA_KEY:-}" ]]; then
    params+=" --cluster-signing-cert-file=${CA_CERT_PATH}"
    params+=" --cluster-signing-key-file=${CA_KEY_PATH}"
  fi
  if [[ -n "${SERVICE_CLUSTER_IP_RANGE:-}" ]]; then
    params+=" --service-cluster-ip-range=${SERVICE_CLUSTER_IP_RANGE}"
  fi
  if [[ -n "${CONCURRENT_SERVICE_SYNCS:-}" ]]; then
    params+=" --concurrent-service-syncs=${CONCURRENT_SERVICE_SYNCS}"
  fi
  if [[ "${NETWORK_PROVIDER:-}" == "kubenet" ]]; then
    params+=" --allocate-node-cidrs=true"
  elif [[ -n "${ALLOCATE_NODE_CIDRS:-}" ]]; then
    params+=" --allocate-node-cidrs=${ALLOCATE_NODE_CIDRS}"
  fi
  if [[ -n "${TERMINATED_POD_GC_THRESHOLD:-}" ]]; then
    params+=" --terminated-pod-gc-threshold=${TERMINATED_POD_GC_THRESHOLD}"
  fi
  if [[ "${ENABLE_IP_ALIASES:-}" == 'true' ]]; then
    params+=" --cidr-allocator-type=${NODE_IPAM_MODE}"
    params+=" --configure-cloud-routes=false"
  fi
  if [[ -n "${FEATURE_GATES:-}" ]]; then
    params+=" --feature-gates=${FEATURE_GATES}"
  fi
  if [[ -n "${VOLUME_PLUGIN_DIR:-}" ]]; then
    params+=" --flex-volume-plugin-dir=${VOLUME_PLUGIN_DIR}"
  fi
  if [[ -n "${CLUSTER_SIGNING_DURATION:-}" ]]; then
    params+=" --experimental-cluster-signing-duration=$CLUSTER_SIGNING_DURATION"
  fi
  # Disable using HPA metrics REST clients if metrics-server isn't enabled,
  # or if we want to explicitly disable it by setting HPA_USE_REST_CLIENT.
  if [[ "${ENABLE_METRICS_SERVER:-}" != "true" ]] ||
     [[ "${HPA_USE_REST_CLIENTS:-}" == "false" ]]; then
    params+=" --horizontal-pod-autoscaler-use-rest-clients=false"
  fi
  if [[ -n "${PV_RECYCLER_OVERRIDE_TEMPLATE:-}" ]]; then
    params+=" --pv-recycler-pod-template-filepath-nfs=$PV_RECYCLER_OVERRIDE_TEMPLATE"
    params+=" --pv-recycler-pod-template-filepath-hostpath=$PV_RECYCLER_OVERRIDE_TEMPLATE"
  fi

  local -r kube_rc_docker_tag=$(cat /home/kubernetes/kube-docker-files/kube-controller-manager.docker_tag)
  local container_env=""
  if [[ -n "${ENABLE_CACHE_MUTATION_DETECTOR:-}" ]]; then
    container_env="\"env\":[{\"name\": \"KUBE_CACHE_MUTATION_DETECTOR\", \"value\": \"${ENABLE_CACHE_MUTATION_DETECTOR}\"}],"
  fi

  local -r src_file="${KUBE_HOME}/kube-manifests/kubernetes/gci-trusty/kube-controller-manager.manifest"
  remove-salt-config-comments "${src_file}"
  # Evaluate variables.
  sed -i -e "s@{{srv_kube_path}}@/etc/srv/kubernetes@g" "${src_file}"
  sed -i -e "s@{{pillar\['kube_docker_registry'\]}}@${DOCKER_REGISTRY}@g" "${src_file}"
  sed -i -e "s@{{pillar\['kube-controller-manager_docker_tag'\]}}@${kube_rc_docker_tag}@g" "${src_file}"
  sed -i -e "s@{{params}}@${params}@g" "${src_file}"
  sed -i -e "s@{{container_env}}@${container_env}@g" ${src_file}
  sed -i -e "s@{{cloud_config_mount}}@${CLOUD_CONFIG_MOUNT}@g" "${src_file}"
  sed -i -e "s@{{cloud_config_volume}}@${CLOUD_CONFIG_VOLUME}@g" "${src_file}"
  sed -i -e "s@{{additional_cloud_config_mount}}@@g" "${src_file}"
  sed -i -e "s@{{additional_cloud_config_volume}}@@g" "${src_file}"
  sed -i -e "s@{{pv_recycler_mount}}@${PV_RECYCLER_MOUNT}@g" "${src_file}"
  sed -i -e "s@{{pv_recycler_volume}}@${PV_RECYCLER_VOLUME}@g" "${src_file}"
  sed -i -e "s@{{flexvolume_hostpath_mount}}@${FLEXVOLUME_HOSTPATH_MOUNT}@g" "${src_file}"
  sed -i -e "s@{{flexvolume_hostpath}}@${FLEXVOLUME_HOSTPATH_VOLUME}@g" "${src_file}"

  cp "${src_file}" /etc/kubernetes/manifests
}

# Starts kubernetes scheduler.
# It prepares the log file, loads the docker image, calculates variables, sets them
# in the manifest file, and then copies the manifest file to /etc/kubernetes/manifests.
#
# Assumed vars (which are calculated in compute-master-manifest-variables)
#   DOCKER_REGISTRY
function start-kube-scheduler {
  echo "Start kubernetes scheduler"
  create-kubescheduler-kubeconfig
  prepare-log-file /var/log/kube-scheduler.log

  # Calculate variables and set them in the manifest.
  params="${SCHEDULER_TEST_LOG_LEVEL:-"--v=2"} ${SCHEDULER_TEST_ARGS:-}"
  params+=" --kubeconfig=/etc/srv/kubernetes/kube-scheduler/kubeconfig"
  if [[ -n "${FEATURE_GATES:-}" ]]; then
    params+=" --feature-gates=${FEATURE_GATES}"
  fi
  if [[ -n "${SCHEDULING_ALGORITHM_PROVIDER:-}"  ]]; then
    params+=" --algorithm-provider=${SCHEDULING_ALGORITHM_PROVIDER}"
  fi
  if [[ -n "${SCHEDULER_POLICY_CONFIG:-}" ]]; then
    create-kubescheduler-policy-config
    params+=" --use-legacy-policy-config"
    params+=" --policy-config-file=/etc/srv/kubernetes/kube-scheduler/policy-config"
  fi
  local -r kube_scheduler_docker_tag=$(cat "${KUBE_HOME}/kube-docker-files/kube-scheduler.docker_tag")

  # Remove salt comments and replace variables with values.
  local -r src_file="${KUBE_HOME}/kube-manifests/kubernetes/gci-trusty/kube-scheduler.manifest"
  remove-salt-config-comments "${src_file}"

  sed -i -e "s@{{srv_kube_path}}@/etc/srv/kubernetes@g" "${src_file}"
  sed -i -e "s@{{params}}@${params}@g" "${src_file}"
  sed -i -e "s@{{pillar\['kube_docker_registry'\]}}@${DOCKER_REGISTRY}@g" "${src_file}"
  sed -i -e "s@{{pillar\['kube-scheduler_docker_tag'\]}}@${kube_scheduler_docker_tag}@g" "${src_file}"
  cp "${src_file}" /etc/kubernetes/manifests
}

# Starts cluster autoscaler.
# Assumed vars (which are calculated in function compute-master-manifest-variables)
#   CLOUD_CONFIG_OPT
#   CLOUD_CONFIG_VOLUME
#   CLOUD_CONFIG_MOUNT
function start-cluster-autoscaler {
  if [[ "${ENABLE_CLUSTER_AUTOSCALER:-}" == "true" ]]; then
    echo "Start kubernetes cluster autoscaler"
    prepare-log-file /var/log/cluster-autoscaler.log

    # Remove salt comments and replace variables with values
    local -r src_file="${KUBE_HOME}/kube-manifests/kubernetes/gci-trusty/cluster-autoscaler.manifest"
    remove-salt-config-comments "${src_file}"

    local params="${AUTOSCALER_MIG_CONFIG} ${CLOUD_CONFIG_OPT} ${AUTOSCALER_EXPANDER_CONFIG:---expander=price}"
    sed -i -e "s@{{params}}@${params}@g" "${src_file}"
    sed -i -e "s@{{cloud_config_mount}}@${CLOUD_CONFIG_MOUNT}@g" "${src_file}"
    sed -i -e "s@{{cloud_config_volume}}@${CLOUD_CONFIG_VOLUME}@g" "${src_file}"
    sed -i -e "s@{%.*%}@@g" "${src_file}"

    cp "${src_file}" /etc/kubernetes/manifests
  fi
}

# A helper function for setting up addon manifests.
#
# $1: addon category under /etc/kubernetes
# $2: manifest source dir
# $3: (optional) auxilary manifest source dir
function setup-addon-manifests {
  local -r src_dir="${KUBE_HOME}/kube-manifests/kubernetes/gci-trusty"
  local -r dst_dir="/etc/kubernetes/$1/$2"

  copy-manifests "${src_dir}/$2" "${dst_dir}"

  # If the PodSecurityPolicy admission controller is enabled,
  # set up the corresponding addon policies.
  if [[ "${ENABLE_POD_SECURITY_POLICY:-}" == "true" ]]; then
    local -r psp_dir="${src_dir}/${3:-$2}/podsecuritypolicies"
    if [[ -d "${psp_dir}" ]]; then
      copy-manifests "${psp_dir}" "${dst_dir}"
    fi
  fi
}

# A helper function for copying manifests and setting dir/files
# permissions.
#
# $1: absolute source dir
# $2: absolute destination dir
function copy-manifests {
  local -r src_dir="$1"
  local -r dst_dir="$2"
  if [[ ! -d "${dst_dir}" ]]; then
    mkdir -p "${dst_dir}"
  fi
  local files=$(find "${src_dir}" -maxdepth 1 -name "*.yaml")
  if [[ -n "${files}" ]]; then
    cp "${src_dir}/"*.yaml "${dst_dir}"
  fi
  files=$(find "${src_dir}" -maxdepth 1 -name "*.json")
  if [[ -n "${files}" ]]; then
    cp "${src_dir}/"*.json "${dst_dir}"
  fi
  files=$(find "${src_dir}" -maxdepth 1 -name "*.yaml.in")
  if [[ -n "${files}" ]]; then
    cp "${src_dir}/"*.yaml.in "${dst_dir}"
  fi
  chown -R root:root "${dst_dir}"
  chmod 755 "${dst_dir}"
  chmod 644 "${dst_dir}"/*
}

# Fluentd manifest is modified using kubectl, which may not be available at
# this point. Run this as a background process.
function wait-for-apiserver-and-update-fluentd {
  local -r fluentd_gcp_yaml="${1}"

  local modifying_flags=""
  if [[ -n "${FLUENTD_GCP_MEMORY_LIMIT:-}" ]]; then
    modifying_flags="${modifying_flags} --limits=memory=${FLUENTD_GCP_MEMORY_LIMIT}"
  fi
  local request_resources=""
  if [[ -n "${FLUENTD_GCP_CPU_REQUEST:-}" ]]; then
    request_resources="cpu=${FLUENTD_GCP_CPU_REQUEST}"
  fi
  if [[ -n "${FLUENTD_GCP_MEMORY_REQUEST:-}" ]]; then
    if [[ -n "${request_resources}" ]]; then
      request_resources="${request_resources},"
    fi
    request_resources="memory=${FLUENTD_GCP_MEMORY_REQUEST}"
  fi
  if [[ -n "${request_resources}" ]]; then
    modifying_flags="${modifying_flags} --requests=${request_resources}"
  fi

  until kubectl get nodes
  do
    sleep 10
  done

  local -r temp_fluentd_gcp_yaml="${fluentd_gcp_yaml}.tmp"
  if kubectl set resources --dry-run --local -f ${fluentd_gcp_yaml} ${modifying_flags} \
      --containers=fluentd-gcp -o yaml > ${temp_fluentd_gcp_yaml}; then
    mv ${temp_fluentd_gcp_yaml} ${fluentd_gcp_yaml}
  else
    (echo "Failed to update fluentd resources. Used manifest:" && cat ${temp_fluentd_gcp_yaml}) >&2
    rm ${temp_fluentd_gcp_yaml}
  fi
}

# Trigger background process that will ultimately update fluentd resource
# requirements.
function start-fluentd-resource-update {
  local -r fluentd_gcp_yaml="${1}"

  wait-for-apiserver-and-update-fluentd ${fluentd_gcp_yaml} &
}

# Updates parameters in yaml file for prometheus-to-sd configuration, or
# removes component if it is disabled.
function update-prometheus-to-sd-parameters {
  if [[ "${ENABLE_PROMETHEUS_TO_SD:-}" == "true" ]]; then
    sed -i -e "s@{{ *prometheus_to_sd_prefix *}}@${PROMETHEUS_TO_SD_PREFIX}@g" "$1"
    sed -i -e "s@{{ *prometheus_to_sd_endpoint *}}@${PROMETHEUS_TO_SD_ENDPOINT}@g" "$1"
  else
    # Removes all lines between two patterns (throws away prometheus-to-sd)
    sed -i -e "/# BEGIN_PROMETHEUS_TO_SD/,/# END_PROMETHEUS_TO_SD/d" "$1"
   fi
}

function update-dashboard-controller {
  if [ -n "${CUSTOM_KUBE_DASHBOARD_BANNER:-}" ]; then
    sed -i -e "s@\( \+\)# PLATFORM-SPECIFIC ARGS HERE@\1- --system-banner=${CUSTOM_KUBE_DASHBOARD_BANNER}\n\1- --system-banner-severity=WARNING@" "$1"
  fi
}

# Sets up the manifests of coreDNS for k8s addons.
function setup-coredns-manifest {
  local -r coredns_file="${dst_dir}/dns/coredns.yaml"
  mv "${dst_dir}/dns/coredns.yaml.in" "${coredns_file}"
  # Replace the salt configurations with variable values.
  sed -i -e "s@{{ *pillar\['dns_domain'\] *}}@${DNS_DOMAIN}@g" "${coredns_file}"
  sed -i -e "s@{{ *pillar\['dns_server'\] *}}@${DNS_SERVER_IP}@g" "${coredns_file}"
  sed -i -e "s@{{ *pillar\['service_cluster_ip_range'\] *}}@${SERVICE_CLUSTER_IP_RANGE}@g" "${coredns_file}"
}

# Sets up the manifests of kube-dns for k8s addons.
function setup-kube-dns-manifest {
  local -r kubedns_file="${dst_dir}/dns/kube-dns.yaml"
  mv "${dst_dir}/dns/kube-dns.yaml.in" "${kubedns_file}"
  if [ -n "${CUSTOM_KUBE_DNS_YAML:-}" ]; then
    # Replace with custom GKE kube-dns deployment.
    cat > "${kubedns_file}" <<EOF
$(echo "$CUSTOM_KUBE_DNS_YAML")
EOF
    update-prometheus-to-sd-parameters ${kubedns_file}
  fi
  # Replace the salt configurations with variable values.
  sed -i -e "s@{{ *pillar\['dns_domain'\] *}}@${DNS_DOMAIN}@g" "${kubedns_file}"
  sed -i -e "s@{{ *pillar\['dns_server'\] *}}@${DNS_SERVER_IP}@g" "${kubedns_file}"

  if [[ "${ENABLE_DNS_HORIZONTAL_AUTOSCALER:-}" == "true" ]]; then
    setup-addon-manifests "addons" "dns-horizontal-autoscaler"
  fi
}

# Prepares the manifests of k8s addons, and starts the addon manager.
# Vars assumed:
#   CLUSTER_NAME
function start-kube-addons {
  echo "Prepare kube-addons manifests and start kube addon manager"
  local -r src_dir="${KUBE_HOME}/kube-manifests/kubernetes/gci-trusty"
  local -r dst_dir="/etc/kubernetes/addons"

  # prep addition kube-up specific rbac objects
  setup-addon-manifests "addons" "rbac/kubelet-api-auth"
  setup-addon-manifests "addons" "rbac/kubelet-cert-rotation"
  if [[ "${REGISTER_MASTER_KUBELET:-false}" == "true" ]]; then
    setup-addon-manifests "addons" "rbac/legacy-kubelet-user"
  else
    setup-addon-manifests "addons" "rbac/legacy-kubelet-user-disable"
  fi

  if [[ "${ENABLE_POD_SECURITY_POLICY:-}" == "true" ]]; then
    setup-addon-manifests "addons" "podsecuritypolicies"
  fi

  # Set up manifests of other addons.
  if [[ "${KUBE_PROXY_DAEMONSET:-}" == "true" ]]; then
    if [ -n "${CUSTOM_KUBE_PROXY_YAML:-}" ]; then
      # Replace with custom GKE kube proxy.
      cat > "$src_dir/kube-proxy/kube-proxy-ds.yaml" <<EOF
$(echo "$CUSTOM_KUBE_PROXY_YAML")
EOF
      update-prometheus-to-sd-parameters "$src_dir/kube-proxy/kube-proxy-ds.yaml"
    fi
    prepare-kube-proxy-manifest-variables "$src_dir/kube-proxy/kube-proxy-ds.yaml"
    setup-addon-manifests "addons" "kube-proxy"
  fi
  if [[ "${ENABLE_CLUSTER_MONITORING:-}" == "influxdb" ]] || \
     [[ "${ENABLE_CLUSTER_MONITORING:-}" == "google" ]] || \
     [[ "${ENABLE_CLUSTER_MONITORING:-}" == "stackdriver" ]] || \
     [[ "${ENABLE_CLUSTER_MONITORING:-}" == "standalone" ]] || \
     [[ "${ENABLE_CLUSTER_MONITORING:-}" == "googleinfluxdb" ]]; then
    local -r file_dir="cluster-monitoring/${ENABLE_CLUSTER_MONITORING}"
    setup-addon-manifests "addons" "cluster-monitoring"
    setup-addon-manifests "addons" "${file_dir}"
    # Replace the salt configurations with variable values.
    base_metrics_memory="${HEAPSTER_GCP_BASE_MEMORY:-140Mi}"
    base_eventer_memory="190Mi"
    base_metrics_cpu="${HEAPSTER_GCP_BASE_CPU:-80m}"
    nanny_memory="90Mi"
    local -r metrics_memory_per_node="${HEAPSTER_GCP_MEMORY_PER_NODE:-4}"
    local -r metrics_cpu_per_node="${HEAPSTER_GCP_CPU_PER_NODE:-0.5}"
    local -r eventer_memory_per_node="500"
    local -r nanny_memory_per_node="200"
    if [[ -n "${NUM_NODES:-}" && "${NUM_NODES}" -ge 1 ]]; then
      num_kube_nodes="$((${NUM_NODES}+1))"
      nanny_memory="$((${num_kube_nodes} * ${nanny_memory_per_node} + 90 * 1024))Ki"
    fi
    controller_yaml="${dst_dir}/${file_dir}"
    if [[ "${ENABLE_CLUSTER_MONITORING:-}" == "googleinfluxdb" ]]; then
      controller_yaml="${controller_yaml}/heapster-controller-combined.yaml"
    else
      controller_yaml="${controller_yaml}/heapster-controller.yaml"
    fi
    remove-salt-config-comments "${controller_yaml}"

    sed -i -e "s@{{ cluster_name }}@${CLUSTER_NAME}@g" "${controller_yaml}"
    sed -i -e "s@{{ *base_metrics_memory *}}@${base_metrics_memory}@g" "${controller_yaml}"
    sed -i -e "s@{{ *base_metrics_cpu *}}@${base_metrics_cpu}@g" "${controller_yaml}"
    sed -i -e "s@{{ *base_eventer_memory *}}@${base_eventer_memory}@g" "${controller_yaml}"
    sed -i -e "s@{{ *metrics_memory_per_node *}}@${metrics_memory_per_node}@g" "${controller_yaml}"
    sed -i -e "s@{{ *eventer_memory_per_node *}}@${eventer_memory_per_node}@g" "${controller_yaml}"
    sed -i -e "s@{{ *nanny_memory *}}@${nanny_memory}@g" "${controller_yaml}"
    sed -i -e "s@{{ *metrics_cpu_per_node *}}@${metrics_cpu_per_node}@g" "${controller_yaml}"
    update-prometheus-to-sd-parameters ${controller_yaml}

    if [[ "${ENABLE_CLUSTER_MONITORING:-}" == "stackdriver" ]]; then
      use_old_resources="${HEAPSTER_USE_OLD_STACKDRIVER_RESOURCES:-true}"
      use_new_resources="${HEAPSTER_USE_NEW_STACKDRIVER_RESOURCES:-false}"
      sed -i -e "s@{{ use_old_resources }}@${use_old_resources}@g" "${controller_yaml}"
      sed -i -e "s@{{ use_new_resources }}@${use_new_resources}@g" "${controller_yaml}"
    fi
  fi
  if [[ "${ENABLE_CLUSTER_MONITORING:-}" == "stackdriver" ]] ||
     ([[ "${ENABLE_CLUSTER_LOGGING:-}" == "true" ]] &&
     [[ "${LOGGING_DESTINATION:-}" == "gcp" ]]); then
    if [[ "${ENABLE_METADATA_AGENT:-}" == "stackdriver" ]] &&
       [[ "${METADATA_AGENT_VERSION:-}" != "" ]]; then
      metadata_agent_cpu_request="${METADATA_AGENT_CPU_REQUEST:-40m}"
      metadata_agent_memory_request="${METADATA_AGENT_MEMORY_REQUEST:-50Mi}"
      setup-addon-manifests "addons" "metadata-agent/stackdriver"
      daemon_set_yaml="${dst_dir}/metadata-agent/stackdriver/metadata-agent.yaml"
      sed -i -e "s@{{ metadata_agent_version }}@${METADATA_AGENT_VERSION}@g" "${daemon_set_yaml}"
      sed -i -e "s@{{ metadata_agent_cpu_request }}@${metadata_agent_cpu_request}@g" "${daemon_set_yaml}"
      sed -i -e "s@{{ metadata_agent_memory_request }}@${metadata_agent_memory_request}@g" "${daemon_set_yaml}"
    fi
  fi
  if [[ "${ENABLE_METRICS_SERVER:-}" == "true" ]]; then
    setup-addon-manifests "addons" "metrics-server"
  fi
  if [[ "${ENABLE_NVIDIA_GPU_DEVICE_PLUGIN:-}" == "true" ]]; then
    setup-addon-manifests "addons" "device-plugins/nvidia-gpu"
  fi
  if [[ "${ENABLE_CLUSTER_DNS:-}" == "true" ]]; then
    setup-addon-manifests "addons" "dns"
    if [[ "${CLUSTER_DNS_CORE_DNS:-}" == "true" ]]; then
      setup-coredns-manifest
    else
      setup-kube-dns-manifest
    fi
  fi
  if [[ "${ENABLE_CLUSTER_REGISTRY:-}" == "true" ]]; then
    setup-addon-manifests "addons" "registry"
    local -r registry_pv_file="${dst_dir}/registry/registry-pv.yaml"
    local -r registry_pvc_file="${dst_dir}/registry/registry-pvc.yaml"
    mv "${dst_dir}/registry/registry-pv.yaml.in" "${registry_pv_file}"
    mv "${dst_dir}/registry/registry-pvc.yaml.in" "${registry_pvc_file}"
    # Replace the salt configurations with variable values.
    remove-salt-config-comments "${controller_yaml}"
    sed -i -e "s@{{ *pillar\['cluster_registry_disk_size'\] *}}@${CLUSTER_REGISTRY_DISK_SIZE}@g" "${registry_pv_file}"
    sed -i -e "s@{{ *pillar\['cluster_registry_disk_size'\] *}}@${CLUSTER_REGISTRY_DISK_SIZE}@g" "${registry_pvc_file}"
    sed -i -e "s@{{ *pillar\['cluster_registry_disk_name'\] *}}@${CLUSTER_REGISTRY_DISK}@g" "${registry_pvc_file}"
  fi
  if [[ "${ENABLE_NODE_LOGGING:-}" == "true" ]] && \
     [[ "${LOGGING_DESTINATION:-}" == "elasticsearch" ]] && \
     [[ "${ENABLE_CLUSTER_LOGGING:-}" == "true" ]]; then
    setup-addon-manifests "addons" "fluentd-elasticsearch"
  fi
  if [[ "${ENABLE_NODE_LOGGING:-}" == "true" ]] && \
     [[ "${LOGGING_DESTINATION:-}" == "gcp" ]]; then
    setup-addon-manifests "addons" "fluentd-gcp"
    local -r event_exporter_yaml="${dst_dir}/fluentd-gcp/event-exporter.yaml"
    local -r fluentd_gcp_yaml="${dst_dir}/fluentd-gcp/fluentd-gcp-ds.yaml"
    update-prometheus-to-sd-parameters ${event_exporter_yaml}
    update-prometheus-to-sd-parameters ${fluentd_gcp_yaml}
    start-fluentd-resource-update ${fluentd_gcp_yaml}
  fi
  if [[ "${ENABLE_CLUSTER_UI:-}" == "true" ]]; then
    setup-addon-manifests "addons" "dashboard"
    local -r dashboard_controller_yaml="${dst_dir}/dashboard/dashboard-controller.yaml"
    update-dashboard-controller ${dashboard_controller_yaml}
  fi
  if [[ "${ENABLE_NODE_PROBLEM_DETECTOR:-}" == "daemonset" ]]; then
    setup-addon-manifests "addons" "node-problem-detector"
  fi
  if [[ "${ENABLE_NODE_PROBLEM_DETECTOR:-}" == "standalone" ]]; then
    # Setup role binding for standalone node problem detector.
    setup-addon-manifests "addons" "node-problem-detector/standalone" "node-problem-detector"
  fi
  if echo "${ADMISSION_CONTROL:-}" | grep -q "LimitRanger"; then
    setup-addon-manifests "admission-controls" "limit-range"
  fi
  if [[ "${NETWORK_POLICY_PROVIDER:-}" == "calico" ]]; then
    setup-addon-manifests "addons" "calico-policy-controller"

    # Configure Calico CNI directory.
    local -r ds_file="${dst_dir}/calico-policy-controller/calico-node-daemonset.yaml"
    sed -i -e "s@__CALICO_CNI_DIR__@/home/kubernetes/bin@g" "${ds_file}"
  fi
  if [[ "${ENABLE_DEFAULT_STORAGE_CLASS:-}" == "true" ]]; then
    setup-addon-manifests "addons" "storage-class/gce"
  fi
  if [[ "${ENABLE_IP_MASQ_AGENT:-}" == "true" ]]; then
    setup-addon-manifests "addons" "ip-masq-agent"
  fi
  if [[ "${ENABLE_METADATA_CONCEALMENT:-}" == "true" ]]; then
    setup-addon-manifests "addons" "metadata-proxy/gce"
    local -r metadata_proxy_yaml="${dst_dir}/metadata-proxy/gce/metadata-proxy.yaml"
    update-prometheus-to-sd-parameters ${metadata_proxy_yaml}
  fi

  # Place addon manager pod manifest.
  cp "${src_dir}/kube-addon-manager.yaml" /etc/kubernetes/manifests
}

# Starts an image-puller - used in test clusters.
function start-image-puller {
  echo "Start image-puller"
  cp "${KUBE_HOME}/kube-manifests/kubernetes/gci-trusty/e2e-image-puller.manifest" \
    /etc/kubernetes/manifests/
}

# Starts kube-registry proxy
function start-kube-registry-proxy {
  echo "Start kube-registry-proxy"
  cp "${KUBE_HOME}/kube-manifests/kubernetes/kube-registry-proxy.yaml" /etc/kubernetes/manifests
}

# Starts a l7 loadbalancing controller for ingress.
function start-lb-controller {
  if [[ "${ENABLE_L7_LOADBALANCING:-}" == "glbc" ]]; then
    echo "Start GCE L7 pod"
    prepare-log-file /var/log/glbc.log
    setup-addon-manifests "addons" "cluster-loadbalancing/glbc"

    local -r glbc_manifest="${KUBE_HOME}/kube-manifests/kubernetes/gci-trusty/glbc.manifest"
    if [[ ! -z "${GCE_GLBC_IMAGE:-}" ]]; then
      sed -i "s@image:.*@image: ${GCE_GLBC_IMAGE}@" "${glbc_manifest}"
    fi
    cp "${glbc_manifest}" /etc/kubernetes/manifests/
  fi
}

# Starts rescheduler.
function start-rescheduler {
  if [[ "${ENABLE_RESCHEDULER:-}" == "true" ]]; then
    echo "Start Rescheduler"
    prepare-log-file /var/log/rescheduler.log
    cp "${KUBE_HOME}/kube-manifests/kubernetes/gci-trusty/rescheduler.manifest" \
       /etc/kubernetes/manifests/
  fi
}

# Setup working directory for kubelet.
function setup-kubelet-dir {
    echo "Making /var/lib/kubelet executable for kubelet"
    mount -B /var/lib/kubelet /var/lib/kubelet/
    mount -B -o remount,exec,suid,dev /var/lib/kubelet
}

function reset-motd {
  # kubelet is installed both on the master and nodes, and the version is easy to parse (unlike kubectl)
  local -r version="$("${KUBE_HOME}"/bin/kubelet --version=true | cut -f2 -d " ")"
  # This logic grabs either a release tag (v1.2.1 or v1.2.1-alpha.1),
  # or the git hash that's in the build info.
  local gitref="$(echo "${version}" | sed -r "s/(v[0-9]+\.[0-9]+\.[0-9]+)(-[a-z]+\.[0-9]+)?.*/\1\2/g")"
  local devel=""
  if [[ "${gitref}" != "${version}" ]]; then
    devel="
Note: This looks like a development version, which might not be present on GitHub.
If it isn't, the closest tag is at:
  https://github.com/kubernetes/kubernetes/tree/${gitref}
"
    gitref="${version//*+/}"
  fi
  cat > /etc/motd <<EOF

Welcome to Kubernetes ${version}!

You can find documentation for Kubernetes at:
  http://docs.kubernetes.io/

The source for this release can be found at:
  /home/kubernetes/kubernetes-src.tar.gz
Or you can download it at:
  https://storage.googleapis.com/kubernetes-release/release/${version}/kubernetes-src.tar.gz

It is based on the Kubernetes source at:
  https://github.com/kubernetes/kubernetes/tree/${gitref}
${devel}
For Kubernetes copyright and licensing information, see:
  /home/kubernetes/LICENSES

EOF
}

function override-kubectl {
    echo "overriding kubectl"
    echo "export PATH=${KUBE_HOME}/bin:\$PATH" > /etc/profile.d/kube_env.sh
}

function override-pv-recycler {
  if [[ -z "${PV_RECYCLER_OVERRIDE_TEMPLATE:-}" ]]; then
    echo "PV_RECYCLER_OVERRIDE_TEMPLATE is not set"
    exit 1
  fi

  PV_RECYCLER_VOLUME="{\"name\": \"pv-recycler-mount\",\"hostPath\": {\"path\": \"${PV_RECYCLER_OVERRIDE_TEMPLATE}\", \"type\": \"FileOrCreate\"}},"
  PV_RECYCLER_MOUNT="{\"name\": \"pv-recycler-mount\",\"mountPath\": \"${PV_RECYCLER_OVERRIDE_TEMPLATE}\", \"readOnly\": true},"

  cat > ${PV_RECYCLER_OVERRIDE_TEMPLATE} <<EOF
version: v1
kind: Pod
metadata:
  generateName: pv-recycler-
  namespace: default
spec:
  activeDeadlineSeconds: 60
  restartPolicy: Never
  volumes:
  - name: vol
  containers:
  - name: pv-recycler
    image: gcr.io/google_containers/busybox:1.27
    command:
    - /bin/sh
    args:
    - -c
    - test -e /scrub && rm -rf /scrub/..?* /scrub/.[!.]* /scrub/* && test -z $(ls -A /scrub) || exit 1
    volumeMounts:
    - name: vol
      mountPath: /scrub
EOF
}

########### Main Function ###########
echo "Start to configure instance for kubernetes"

KUBE_HOME="/home/kubernetes"
CONTAINERIZED_MOUNTER_HOME="${KUBE_HOME}/containerized_mounter"
PV_RECYCLER_OVERRIDE_TEMPLATE="${KUBE_HOME}/kube-manifests/kubernetes/pv-recycler-template.yaml"

if [[ ! -e "${KUBE_HOME}/kube-env" ]]; then
  echo "The ${KUBE_HOME}/kube-env file does not exist!! Terminate cluster initialization."
  exit 1
fi

source "${KUBE_HOME}/kube-env"

if [[ -e "${KUBE_HOME}/kube-master-certs" ]]; then
  source "${KUBE_HOME}/kube-master-certs"
fi

if [[ -n "${KUBE_USER:-}" ]]; then
  if ! [[ "${KUBE_USER}" =~ ^[-._@a-zA-Z0-9]+$ ]]; then
    echo "Bad KUBE_USER format."
    exit 1
  fi
fi

# generate the controller manager and scheduler tokens here since they are only used on the master.
KUBE_CONTROLLER_MANAGER_TOKEN=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
KUBE_SCHEDULER_TOKEN=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)

setup-os-params
config-ip-firewall
create-dirs
setup-kubelet-dir
ensure-local-ssds
setup-logrotate
if [[ "${KUBERNETES_MASTER:-}" == "true" ]]; then
  mount-master-pd
  create-node-pki
  create-master-pki
  create-master-auth
  create-master-kubelet-auth
  create-master-etcd-auth
  override-pv-recycler
else
  create-node-pki
  create-kubelet-kubeconfig ${KUBERNETES_MASTER_NAME}
  if [[ "${KUBE_PROXY_DAEMONSET:-}" != "true" ]]; then
    create-kubeproxy-user-kubeconfig
  fi
  if [[ "${ENABLE_NODE_PROBLEM_DETECTOR:-}" == "standalone" ]]; then
    create-node-problem-detector-kubeconfig
  fi
fi

override-kubectl
# Run the containerized mounter once to pre-cache the container image.
if [[ "${CONTAINER_RUNTIME:-docker}" == "docker" ]]; then
  assemble-docker-flags
fi
start-kubelet

if [[ "${KUBERNETES_MASTER:-}" == "true" ]]; then
  compute-master-manifest-variables
  start-etcd-servers
  start-etcd-empty-dir-cleanup-pod
  start-kube-apiserver
  start-kube-controller-manager
  start-kube-scheduler
  start-kube-addons
  start-cluster-autoscaler
  start-lb-controller
  start-rescheduler
else
  if [[ "${KUBE_PROXY_DAEMONSET:-}" != "true" ]]; then
    start-kube-proxy
  fi
  # Kube-registry-proxy.
  if [[ "${ENABLE_CLUSTER_REGISTRY:-}" == "true" ]]; then
    start-kube-registry-proxy
  fi
  if [[ "${PREPULL_E2E_IMAGES:-}" == "true" ]]; then
    start-image-puller
  fi
  if [[ "${ENABLE_NODE_PROBLEM_DETECTOR:-}" == "standalone" ]]; then
    start-node-problem-detector
  fi
fi
reset-motd
prepare-mounter-rootfs
modprobe configs
echo "Done for the configuration for kubernetes"
