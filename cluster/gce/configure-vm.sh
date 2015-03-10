#!/bin/bash

# Copyright 2015 Google Inc. All rights reserved.
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

set -o errexit
set -o nounset
set -o pipefail

# If we have any arguments at all, this is a push and not just setup.
is_push=$@

function ensure-install-dir() {
  INSTALL_DIR="/var/cache/kubernetes-install"
  mkdir -p ${INSTALL_DIR}
  cd ${INSTALL_DIR}
}

function set-broken-motd() {
  echo -e '\nBroken (or in progress) GCE Kubernetes node setup! Suggested first step:\n  tail /var/log/startupscript.log\n' > /etc/motd
}

function set-good-motd() {
  echo -e '\n=== GCE Kubernetes node setup complete ===\n' > /etc/motd
}

function curl-metadata() {
  curl --fail --silent -H 'Metadata-Flavor: Google' "http://metadata/computeMetadata/v1/instance/attributes/${1}"
}

function set-kube-env() {
  local kube_env_yaml="${INSTALL_DIR}/kube_env.yaml"

  until curl-metadata kube-env > "${kube_env_yaml}"; do
    echo 'Waiting for kube-env...'
    sleep 3
  done

  # kube-env has all the environment variables we care about, in a flat yaml format
  eval $(python -c '''
import pipes,sys,yaml

for k,v in yaml.load(sys.stdin).iteritems():
  print "readonly {var}={value}".format(var = k, value = pipes.quote(str(v)))
''' < "${kube_env_yaml}")

  # We bake the KUBELET_TOKEN in separately to avoid auth information
  # having to be re-communicated on kube-push. (Otherwise the client
  # has to keep the bearer token around to handle generating a valid
  # kube-env.)
  if [[ -z "${KUBELET_TOKEN:-}" ]]; then
    until KUBELET_TOKEN=$(curl-metadata kube-token); do
      echo 'Waiting for metadata KUBELET_TOKEN...'
      sleep 3
    done
  fi

  if [[ "${KUBERNETES_MASTER}" == "true" ]]; then
    # TODO(zmerlynn): This block of code should disappear once #4561 & #4562 are done
    if [[ -z "${KUBERNETES_NODE_NAMES:-}" ]]; then
      until KUBERNETES_NODE_NAMES=$(curl-metadata kube-node-names); do
        echo 'Waiting for metadata KUBERNETES_NODE_NAMES...'
        sleep 3
      done
    fi
  else
    # And this should go away once the master can allocate CIDRs
    if [[ -z "${MINION_IP_RANGE:-}" ]]; then
      until MINION_IP_RANGE=$(curl-metadata node-ip-range); do
        echo 'Waiting for metadata MINION_IP_RANGE...'
        sleep 3
      done
    fi
  fi
}

function remove-docker-artifacts() {
  # Remove docker artifacts on minion nodes, if present
  iptables -t nat -F || true
  ifconfig docker0 down || true
  brctl delbr docker0 || true
}

# Retry a download until we get it.
#
# $1 is the URL to download
download-or-bust() {
  local -r url="$1"
  local -r file="${url##*/}"
  rm -f "$file"
  until [[ -e "${1##*/}" ]]; do
    echo "Downloading file ($1)"
    curl --ipv4 -Lo "$file" --connect-timeout 20 --retry 6 --retry-delay 10 "$1"
  done
}

# Install salt from GCS.  See README.md for instructions on how to update these
# debs.
install-salt() {
  apt-get update

  mkdir -p /var/cache/salt-install
  cd /var/cache/salt-install

  TARS=(
    libzmq3_3.2.3+dfsg-1~bpo70~dst+1_amd64.deb
    python-zmq_13.1.0-1~bpo70~dst+1_amd64.deb
    salt-common_2014.1.13+ds-1~bpo70+1_all.deb
    salt-minion_2014.1.13+ds-1~bpo70+1_all.deb
  )
  URL_BASE="https://storage.googleapis.com/kubernetes-release/salt"

  for tar in "${TARS[@]}"; do
    download-or-bust "${URL_BASE}/${tar}"
    dpkg -i "${tar}" || true
  done

  # This will install any of the unmet dependencies from above.
  apt-get install -f -y
}

# Ensure salt-minion *isn't* running
stop-salt-minion() {
  # This ensures it on next reboot
  echo manual > /etc/init/salt-minion.override

  service salt-minion stop
  while service salt-minion status >/dev/null; do
    service salt-minion stop # No, really.
    echo "Waiting for salt-minion to shut down"
    sleep 1
  done
}

# Mounts a persistent disk (formatting if needed) to store the persistent data
# on the master -- etcd's data, a few settings, and security certs/keys/tokens.
#
# This function can be reused to mount an existing PD because all of its
# operations modifying the disk are idempotent -- safe_format_and_mount only
# formats an unformatted disk, and mkdir -p will leave a directory be if it
# already exists.
mount-master-pd() {
  device_info=$(ls -l /dev/disk/by-id/google-master-pd)
  relative_path=${device_info##* }
  device_path="/dev/disk/by-id/${relative_path}"

  # Format and mount the disk, create directories on it for all of the master's
  # persistent data, and link them to where they're used.
  mkdir -p /mnt/master-pd
  /usr/share/google/safe_format_and_mount -m "mkfs.ext4 -F" "${device_path}" /mnt/master-pd
  # Contains all the data stored in etcd
  mkdir -m 700 -p /mnt/master-pd/var/etcd
  # Contains the dynamically generated apiserver auth certs and keys
  mkdir -p /mnt/master-pd/srv/kubernetes
  # Contains the cluster's initial config parameters and auth tokens
  mkdir -p /mnt/master-pd/srv/salt-overlay
  ln -s /mnt/master-pd/var/etcd /var/etcd
  ln -s /mnt/master-pd/srv/kubernetes /srv/kubernetes
  ln -s /mnt/master-pd/srv/salt-overlay /srv/salt-overlay

  # This is a bit of a hack to get around the fact that salt has to run after the
  # PD and mounted directory are already set up. We can't give ownership of the
  # directory to etcd until the etcd user and group exist, but they don't exist
  # until salt runs if we don't create them here. We could alternatively make the
  # permissions on the directory more permissive, but this seems less bad.
  useradd -s /sbin/nologin -d /var/etcd etcd
  chown etcd /mnt/master-pd/var/etcd
  chgrp etcd /mnt/master-pd/var/etcd
}

# Create the overlay files for the salt tree.  We create these in a separate
# place so that we can blow away the rest of the salt configs on a kube-push and
# re-apply these.
function create-salt-pillar() {
  # Always overwrite the cluster-params.sls (even on a push, we have
  # these variables)
  mkdir -p /srv/salt-overlay/pillar
  cat <<EOF >/srv/salt-overlay/pillar/cluster-params.sls
instance_prefix: '$(echo "$INSTANCE_PREFIX" | sed -e "s/'/''/g")'
node_instance_prefix: '$(echo "$NODE_INSTANCE_PREFIX" | sed -e "s/'/''/g")'
portal_net: '$(echo "$PORTAL_NET" | sed -e "s/'/''/g")'
enable_cluster_monitoring: '$(echo "$ENABLE_CLUSTER_MONITORING" | sed -e "s/'/''/g")'
enable_node_monitoring: '$(echo "$ENABLE_NODE_MONITORING" | sed -e "s/'/''/g")'
enable_cluster_logging: '$(echo "$ENABLE_CLUSTER_LOGGING" | sed -e "s/'/''/g")'
enable_node_logging: '$(echo "$ENABLE_NODE_LOGGING" | sed -e "s/'/''/g")'
logging_destination: '$(echo "$LOGGING_DESTINATION" | sed -e "s/'/''/g")'
elasticsearch_replicas: '$(echo "$ELASTICSEARCH_LOGGING_REPLICAS" | sed -e "s/'/''/g")'
enable_cluster_dns: '$(echo "$ENABLE_CLUSTER_DNS" | sed -e "s/'/''/g")'
dns_replicas: '$(echo "$DNS_REPLICAS" | sed -e "s/'/''/g")'
dns_server: '$(echo "$DNS_SERVER_IP" | sed -e "s/'/''/g")'
dns_domain: '$(echo "$DNS_DOMAIN" | sed -e "s/'/''/g")'
EOF

  if [[ "${KUBERNETES_MASTER}" == "true" ]]; then
    cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
gce_node_names: '$(echo "$KUBERNETES_NODE_NAMES" | sed -e "s/'/''/g")'
EOF
  fi
}

# This should only happen on cluster initialization
function create-salt-auth() {
  mkdir -p /srv/salt-overlay/salt/nginx
  echo "${MASTER_HTPASSWD}" > /srv/salt-overlay/salt/nginx/htpasswd

  mkdir -p /srv/salt-overlay/salt/kube-apiserver
  known_tokens_file="/srv/salt-overlay/salt/kube-apiserver/known_tokens.csv"
  (umask 077;
    echo "${KUBELET_TOKEN},kubelet,kubelet" > "${known_tokens_file}")

  mkdir -p /srv/salt-overlay/salt/kubelet
  kubelet_auth_file="/srv/salt-overlay/salt/kubelet/kubernetes_auth"
  (umask 077;
    echo "{\"BearerToken\": \"${KUBELET_TOKEN}\", \"Insecure\": true }" > "${kubelet_auth_file}")
}

function download-release() {
  echo "Downloading binary release tar ($SERVER_BINARY_TAR_URL)"
  download-or-bust "$SERVER_BINARY_TAR_URL"

  echo "Downloading binary release tar ($SALT_TAR_URL)"
  download-or-bust "$SALT_TAR_URL"

  echo "Unpacking Salt tree"
  rm -rf kubernetes
  tar xzf "${SALT_TAR_URL##*/}"

  echo "Running release install script"
  sudo kubernetes/saltbase/install.sh "${SERVER_BINARY_TAR_URL##*/}"
}

function fix-apt-sources() {
  sed -i -e "\|^deb.*http://http.debian.net/debian| s/^/#/" /etc/apt/sources.list
  sed -i -e "\|^deb.*http://ftp.debian.org/debian| s/^/#/" /etc/apt/sources.list.d/backports.list
}

function salt-run-local() {
  cat <<EOF >/etc/salt/minion.d/local.conf
file_client: local
file_roots:
  base:
    - /srv/salt
EOF
}

function salt-debug-log() {
  cat <<EOF >/etc/salt/minion.d/log-level-debug.conf
log_level: debug
log_level_logfile: debug
EOF
}

function salt-master-role() {
  cat <<EOF >/etc/salt/minion.d/grains.conf
grains:
  roles:
    - kubernetes-master
  cloud: gce
EOF
}

function salt-node-role() {
  cat <<EOF >/etc/salt/minion.d/grains.conf
grains:
  roles:
    - kubernetes-pool
  cbr-cidr: '$(echo "$MINION_IP_RANGE" | sed -e "s/'/''/g")'
  cloud: gce
EOF
}

function salt-docker-opts() {
  DOCKER_OPTS=""

  if [[ -n "${EXTRA_DOCKER_OPTS-}" ]]; then
    DOCKER_OPTS="${EXTRA_DOCKER_OPTS}"
  fi

  # Decide whether to enable the cache
  if [[ "${ENABLE_DOCKER_REGISTRY_CACHE}" == "true" ]]; then
    REGION=$(echo "${ZONE}" | cut -f 1,2 -d -)
    echo "Enable docker registry cache at region: " $REGION
    DOCKER_OPTS="${DOCKER_OPTS} --registry-mirror='https://${REGION}.docker-cache.clustermaster.net'"
  fi

  if [[ -n "{DOCKER_OPTS}" ]]; then
    cat <<EOF >>/etc/salt/minion.d/grains.conf
  docker_opts: '$(echo "$DOCKER_OPTS" | sed -e "s/'/''/g")'
EOF
  fi
}

function salt-set-apiserver() {
  local kube_master_ip
  until kube_master_ip=$(getent hosts ${KUBERNETES_MASTER_NAME} | cut -f1 -d\ ); do
    echo 'Waiting for DNS resolution of ${KUBERNETES_MASTER_NAME}...'
    sleep 3
  done

  cat <<EOF >>/etc/salt/minion.d/grains.conf
  api_servers: '${kube_master_ip}'
  apiservers: '${kube_master_ip}'
EOF
}

function configure-salt() {
  fix-apt-sources
  mkdir -p /etc/salt/minion.d
  salt-run-local
  if [[ "${KUBERNETES_MASTER}" == "true" ]]; then
    salt-master-role
  else
    salt-node-role
    salt-docker-opts
    salt-set-apiserver
  fi
  install-salt
  stop-salt-minion
}

function run-salt() {
  salt-call --local state.highstate || true
}

####################################################################################

if [[ -z "${is_push}" ]]; then
  echo "== kube-up node config starting =="
  set-broken-motd
  ensure-install-dir
  set-kube-env
  [[ "${KUBERNETES_MASTER}" == "true" ]] && mount-master-pd
  create-salt-pillar
  create-salt-auth
  download-release
  configure-salt
  remove-docker-artifacts
  run-salt
  set-good-motd
  echo "== kube-up node config done =="
else
  echo "== kube-push node config starting =="
  ensure-install-dir
  set-kube-env
  create-salt-pillar
  run-salt
  echo "== kube-push node config done =="
fi
