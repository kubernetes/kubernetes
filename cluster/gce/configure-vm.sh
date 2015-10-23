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

set -o errexit
set -o nounset
set -o pipefail

# If we have any arguments at all, this is a push and not just setup.
is_push=$@

readonly KNOWN_TOKENS_FILE="/srv/salt-overlay/salt/kube-apiserver/known_tokens.csv"
readonly BASIC_AUTH_FILE="/srv/salt-overlay/salt/kube-apiserver/basic_auth.csv"

function ensure-basic-networking() {
  # Deal with GCE networking bring-up race. (We rely on DNS for a lot,
  # and it's just not worth doing a whole lot of startup work if this
  # isn't ready yet.)
  until getent hosts metadata.google.internal &>/dev/null; do
    echo 'Waiting for functional DNS (trying to resolve metadata.google.internal)...'
    sleep 3
  done
  until getent hosts $(hostname -f) &>/dev/null; do
    echo 'Waiting for functional DNS (trying to resolve my own FQDN)...'
    sleep 3
  done
  until getent hosts $(hostname -i) &>/dev/null; do
    echo 'Waiting for functional DNS (trying to resolve my own IP)...'
    sleep 3
  done

  echo "Networking functional on $(hostname) ($(hostname -i))"
}

function ensure-install-dir() {
  INSTALL_DIR="/var/cache/kubernetes-install"
  mkdir -p ${INSTALL_DIR}
  cd ${INSTALL_DIR}
}

function salt-apiserver-timeout-grain() {
    cat <<EOF >>/etc/salt/minion.d/grains.conf
  minRequestTimeout: '$1'
EOF
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
}

function remove-docker-artifacts() {
  echo "== Deleting docker0 =="
  # Forcibly install bridge-utils (options borrowed from Salt logs).
  until apt-get -q -y -o DPkg::Options::=--force-confold -o DPkg::Options::=--force-confdef install bridge-utils; do
    echo "== install of bridge-utils failed, retrying =="
    sleep 5
  done

  # Remove docker artifacts on minion nodes, if present
  iptables -t nat -F || true
  ifconfig docker0 down || true
  brctl delbr docker0 || true
  echo "== Finished deleting docker0 =="
}

# Retry a download until we get it.
#
# $1 is the URL to download
download-or-bust() {
  local -r url="$1"
  local -r file="${url##*/}"
  rm -f "$file"
  until curl --ipv4 -Lo "$file" --connect-timeout 20 --retry 6 --retry-delay 10 "${url}"; do
    echo "Failed to download file (${url}). Retrying."
  done
}

validate-hash() {
  local -r file="$1"
  local -r expected="$2"
  local actual

  actual=$(sha1sum ${file} | awk '{ print $1 }') || true
  if [[ "${actual}" != "${expected}" ]]; then
    echo "== ${file} corrupted, sha1 ${actual} doesn't match expected ${expected} =="
    return 1
  fi
}

# Install salt from GCS.  See README.md for instructions on how to update these
# debs.
install-salt() {
  if dpkg -s salt-minion &>/dev/null; then
    echo "== SaltStack already installed, skipping install step =="
    return
  fi

  echo "== Refreshing package database =="
  until apt-get update; do
    echo "== apt-get update failed, retrying =="
    echo sleep 5
  done

  mkdir -p /var/cache/salt-install
  cd /var/cache/salt-install

  DEBS=(
    libzmq3_3.2.3+dfsg-1~bpo70~dst+1_amd64.deb
    python-zmq_13.1.0-1~bpo70~dst+1_amd64.deb
    salt-common_2014.1.13+ds-1~bpo70+1_all.deb
    salt-minion_2014.1.13+ds-1~bpo70+1_all.deb
  )
  URL_BASE="https://storage.googleapis.com/kubernetes-release/salt"

  for deb in "${DEBS[@]}"; do
    if [ ! -e "${deb}" ]; then
      download-or-bust "${URL_BASE}/${deb}"
    fi
  done

  # Based on
  # https://major.io/2014/06/26/install-debian-packages-without-starting-daemons/
  # We do this to prevent Salt from starting the salt-minion
  # daemon. The other packages don't have relevant daemons. (If you
  # add a package that needs a daemon started, add it to a different
  # list.)
  cat > /usr/sbin/policy-rc.d <<EOF
#!/bin/sh
echo "Salt shall not start." >&2
exit 101
EOF
  chmod 0755 /usr/sbin/policy-rc.d

  for deb in "${DEBS[@]}"; do
    echo "== Installing ${deb}, ignore dependency complaints (will fix later) =="
    dpkg --skip-same-version --force-depends -i "${deb}"
  done

  # This will install any of the unmet dependencies from above.
  echo "== Installing unmet dependencies =="
  until apt-get install -f -y; do
    echo "== apt-get install failed, retrying =="
    echo sleep 5
  done

  rm /usr/sbin/policy-rc.d

  # Log a timestamp
  echo "== Finished installing Salt =="
}

# Ensure salt-minion isn't running and never runs
stop-salt-minion() {
  if [[ -e /etc/init/salt-minion.override ]]; then
    # Assume this has already run (upgrade, or baked into containervm)
    return
  fi

  # This ensures it on next reboot
  echo manual > /etc/init/salt-minion.override
  update-rc.d salt-minion disable

  while service salt-minion status >/dev/null; do
    echo "salt-minion found running, stopping"
    service salt-minion stop
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
  # TODO(zmerlynn): GKE is still lagging in master-pd creation
  if [[ ! -e /dev/disk/by-id/google-master-pd ]]; then
    return
  fi
  device_info=$(ls -l /dev/disk/by-id/google-master-pd)
  relative_path=${device_info##* }
  device_path="/dev/disk/by-id/${relative_path}"

  # Format and mount the disk, create directories on it for all of the master's
  # persistent data, and link them to where they're used.
  echo "Mounting master-pd"
  mkdir -p /mnt/master-pd
  /usr/share/google/safe_format_and_mount -m "mkfs.ext4 -F" "${device_path}" /mnt/master-pd &>/var/log/master-pd-mount.log || \
    { echo "!!! master-pd mount failed, review /var/log/master-pd-mount.log !!!"; return 1; }
  # Contains all the data stored in etcd
  mkdir -m 700 -p /mnt/master-pd/var/etcd
  # Contains the dynamically generated apiserver auth certs and keys
  mkdir -p /mnt/master-pd/srv/kubernetes
  # Contains the cluster's initial config parameters and auth tokens
  mkdir -p /mnt/master-pd/srv/salt-overlay
  # Directory for kube-apiserver to store SSH key (if necessary)
  mkdir -p /mnt/master-pd/srv/sshproxy

  ln -s -f /mnt/master-pd/var/etcd /var/etcd
  ln -s -f /mnt/master-pd/srv/kubernetes /srv/kubernetes
  ln -s -f /mnt/master-pd/srv/sshproxy /srv/sshproxy
  ln -s -f /mnt/master-pd/srv/salt-overlay /srv/salt-overlay

  # This is a bit of a hack to get around the fact that salt has to run after the
  # PD and mounted directory are already set up. We can't give ownership of the
  # directory to etcd until the etcd user and group exist, but they don't exist
  # until salt runs if we don't create them here. We could alternatively make the
  # permissions on the directory more permissive, but this seems less bad.
  if ! id etcd &>/dev/null; then
    useradd -s /sbin/nologin -d /var/etcd etcd
  fi
  chown -R etcd /mnt/master-pd/var/etcd
  chgrp -R etcd /mnt/master-pd/var/etcd
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
cluster_cidr: '$(echo "$CLUSTER_IP_RANGE" | sed -e "s/'/''/g")'
allocate_node_cidrs: '$(echo "$ALLOCATE_NODE_CIDRS" | sed -e "s/'/''/g")'
service_cluster_ip_range: '$(echo "$SERVICE_CLUSTER_IP_RANGE" | sed -e "s/'/''/g")'
enable_cluster_monitoring: '$(echo "$ENABLE_CLUSTER_MONITORING" | sed -e "s/'/''/g")'
enable_cluster_logging: '$(echo "$ENABLE_CLUSTER_LOGGING" | sed -e "s/'/''/g")'
enable_cluster_ui: '$(echo "$ENABLE_CLUSTER_UI" | sed -e "s/'/''/g")'
enable_l7_loadbalancing: '$(echo "$ENABLE_L7_LOADBALANCING" | sed -e "s/'/''/g")'
enable_node_logging: '$(echo "$ENABLE_NODE_LOGGING" | sed -e "s/'/''/g")'
logging_destination: '$(echo "$LOGGING_DESTINATION" | sed -e "s/'/''/g")'
elasticsearch_replicas: '$(echo "$ELASTICSEARCH_LOGGING_REPLICAS" | sed -e "s/'/''/g")'
enable_cluster_dns: '$(echo "$ENABLE_CLUSTER_DNS" | sed -e "s/'/''/g")'
enable_cluster_registry: '$(echo "$ENABLE_CLUSTER_REGISTRY" | sed -e "s/'/''/g")'
dns_replicas: '$(echo "$DNS_REPLICAS" | sed -e "s/'/''/g")'
dns_server: '$(echo "$DNS_SERVER_IP" | sed -e "s/'/''/g")'
dns_domain: '$(echo "$DNS_DOMAIN" | sed -e "s/'/''/g")'
admission_control: '$(echo "$ADMISSION_CONTROL" | sed -e "s/'/''/g")'
enable_manifest_url: '$(echo "$ENABLE_MANIFEST_URL" | sed -e "s/'/''/g")'
manifest_url: '$(echo "$MANIFEST_URL" | sed -e "s/'/''/g")'
manifest_url_header: '$(echo "$MANIFEST_URL_HEADER" | sed -e "s/'/''/g")'
num_nodes: $(echo "${NUM_MINIONS}")
EOF

    if [ -n "${APISERVER_TEST_ARGS:-}" ]; then
      cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
apiserver_test_args: '$(echo "$APISERVER_TEST_ARGS" | sed -e "s/'/''/g")'
EOF
    fi
    if [ -n "${KUBELET_TEST_ARGS:-}" ]; then
      cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
kubelet_test_args: '$(echo "$KUBELET_TEST_ARGS" | sed -e "s/'/''/g")'
EOF
    fi
    if [ -n "${CONTROLLER_MANAGER_TEST_ARGS:-}" ]; then
      cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
controller_manager_test_args: '$(echo "$CONTROLLER_MANAGER_TEST_ARGS" | sed -e "s/'/''/g")'
EOF
    fi
    if [ -n "${SCHEDULER_TEST_ARGS:-}" ]; then
      cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
scheduler_test_args: '$(echo "$SCHEDULER_TEST_ARGS" | sed -e "s/'/''/g")'
EOF
    fi
    if [ -n "${KUBEPROXY_TEST_ARGS:-}" ]; then
      cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
kubeproxy_test_args: '$(echo "$KUBEPROXY_TEST_ARGS" | sed -e "s/'/''/g")'
EOF
    fi
    # TODO: Replace this  with a persistent volume (and create it).
    if [[ "${ENABLE_CLUSTER_REGISTRY}" == true && -n "${CLUSTER_REGISTRY_DISK}" ]]; then
      cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
cluster_registry_disk_type: gce
cluster_registry_disk_size: $(convert-bytes-gce-kube ${CLUSTER_REGISTRY_DISK_SIZE})
cluster_registry_disk_name: ${CLUSTER_REGISTRY_DISK}
EOF
    fi
    if [ -n "${ENABLE_HORIZONTAL_POD_AUTOSCALER:-}" ]; then
      cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
enable_horizontal_pod_autoscaler: '$(echo "$ENABLE_HORIZONTAL_POD_AUTOSCALER" | sed -e "s/'/''/g")'
EOF
    fi
    if [ -n "${ENABLE_DEPLOYMENTS:-}" ]; then
      cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
enable_deployments: '$(echo "$ENABLE_DEPLOYMENTS" | sed -e "s/'/''/g")'
EOF
    fi
}

# The job of this function is simple, but the basic regular expression syntax makes
# this difficult to read. What we want to do is convert from [0-9]+B, KB, KiB, MB, etc
# into [0-9]+, Ki, Mi, Gi, etc.
# This is done in two steps:
#   1. Convert from [0-9]+X?i?B into [0-9]X? (X denotes the prefix, ? means the field
#      is optional.
#   2. Attach an 'i' to the end of the string if we find a letter.
# The two step process is needed to handle the edge case in which we want to convert
# a raw byte count, as the result should be a simple number (e.g. 5B -> 5).
function convert-bytes-gce-kube() {
  local -r storage_space=$1
  echo "${storage_space}" | sed -e 's/^\([0-9]\+\)\([A-Z]\)\?i\?B$/\1\2/g' -e 's/\([A-Z]\)$/\1i/'
}

# This should only happen on cluster initialization.
#
#  - Uses KUBE_PASSWORD and KUBE_USER to generate basic_auth.csv.
#  - Uses KUBE_BEARER_TOKEN, KUBELET_TOKEN, and KUBE_PROXY_TOKEN to generate
#    known_tokens.csv (KNOWN_TOKENS_FILE).
#  - Uses CA_CERT, MASTER_CERT, and MASTER_KEY to populate the SSL credentials
#    for the apiserver.
#  - Optionally uses KUBECFG_CERT and KUBECFG_KEY to store a copy of the client
#    cert credentials.
#
# After the first boot and on upgrade, these files exists on the master-pd
# and should never be touched again (except perhaps an additional service
# account, see NB below.)
function create-salt-master-auth() {
  if [[ ! -e /srv/kubernetes/ca.crt ]]; then
    if  [[ ! -z "${CA_CERT:-}" ]] && [[ ! -z "${MASTER_CERT:-}" ]] && [[ ! -z "${MASTER_KEY:-}" ]]; then
      mkdir -p /srv/kubernetes
      (umask 077;
        echo "${CA_CERT}" | base64 -d > /srv/kubernetes/ca.crt;
        echo "${MASTER_CERT}" | base64 -d > /srv/kubernetes/server.cert;
        echo "${MASTER_KEY}" | base64 -d > /srv/kubernetes/server.key;
        # Kubecfg cert/key are optional and included for backwards compatibility.
        # TODO(roberthbailey): Remove these two lines once GKE no longer requires
        # fetching clients certs from the master VM.
        echo "${KUBECFG_CERT:-}" | base64 -d > /srv/kubernetes/kubecfg.crt;
        echo "${KUBECFG_KEY:-}" | base64 -d > /srv/kubernetes/kubecfg.key)
    fi
  fi
  if [ ! -e "${BASIC_AUTH_FILE}" ]; then
    mkdir -p /srv/salt-overlay/salt/kube-apiserver
    (umask 077;
      echo "${KUBE_PASSWORD},${KUBE_USER},admin" > "${BASIC_AUTH_FILE}")
  fi
  if [ ! -e "${KNOWN_TOKENS_FILE}" ]; then
    mkdir -p /srv/salt-overlay/salt/kube-apiserver
    (umask 077;
      echo "${KUBE_BEARER_TOKEN},admin,admin" > "${KNOWN_TOKENS_FILE}";
      echo "${KUBELET_TOKEN},kubelet,kubelet" >> "${KNOWN_TOKENS_FILE}";
      echo "${KUBE_PROXY_TOKEN},kube_proxy,kube_proxy" >> "${KNOWN_TOKENS_FILE}")

    # Generate tokens for other "service accounts".  Append to known_tokens.
    #
    # NB: If this list ever changes, this script actually has to
    # change to detect the existence of this file, kill any deleted
    # old tokens and add any new tokens (to handle the upgrade case).
    local -r service_accounts=("system:scheduler" "system:controller_manager" "system:logging" "system:monitoring" "system:dns")
    for account in "${service_accounts[@]}"; do
      token=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
      echo "${token},${account},${account}" >> "${KNOWN_TOKENS_FILE}"
    done
  fi
}

# This should happen only on cluster initialization. After the first boot
# and on upgrade, the kubeconfig file exists on the master-pd and should
# never be touched again.
#
#  - Uses KUBELET_CA_CERT (falling back to CA_CERT), KUBELET_CERT, and
#    KUBELET_KEY to generate a kubeconfig file for the kubelet to securely
#    connect to the apiserver.
function create-salt-master-kubelet-auth() {
  # Only configure the kubelet on the master if the required variables are
  # set in the environment.
  if [[ ! -z "${KUBELET_APISERVER:-}" ]] && [[ ! -z "${KUBELET_CERT:-}" ]] && [[ ! -z "${KUBELET_KEY:-}" ]]; then
    create-salt-kubelet-auth
  fi
}

# This should happen both on cluster initialization and node upgrades.
#
#  - Uses KUBELET_CA_CERT (falling back to CA_CERT), KUBELET_CERT, and
#    KUBELET_KEY to generate a kubeconfig file for the kubelet to securely
#    connect to the apiserver.

function create-salt-kubelet-auth() {
  local -r kubelet_kubeconfig_file="/srv/salt-overlay/salt/kubelet/kubeconfig"
  if [ ! -e "${kubelet_kubeconfig_file}" ]; then
    # If there isn't a CA certificate set specifically for the kubelet, use
    # the cluster CA certificate.
    if [[ -z "${KUBELET_CA_CERT:-}" ]]; then
      KUBELET_CA_CERT="${CA_CERT}"
    fi
    mkdir -p /srv/salt-overlay/salt/kubelet
    (umask 077;
      cat > "${kubelet_kubeconfig_file}" <<EOF
apiVersion: v1
kind: Config
users:
- name: kubelet
  user:
    client-certificate-data: ${KUBELET_CERT}
    client-key-data: ${KUBELET_KEY}
clusters:
- name: local
  cluster:
    certificate-authority-data: ${KUBELET_CA_CERT}
contexts:
- context:
    cluster: local
    user: kubelet
  name: service-account-context
current-context: service-account-context
EOF
)
  fi
}

# This should happen both on cluster initialization and node upgrades.
#
#  - Uses the CA_CERT and KUBE_PROXY_TOKEN to generate a kubeconfig file for
#    the kube-proxy to securely connect to the apiserver.
function create-salt-kubeproxy-auth() {
  local -r kube_proxy_kubeconfig_file="/srv/salt-overlay/salt/kube-proxy/kubeconfig"
  if [ ! -e "${kube_proxy_kubeconfig_file}" ]; then
    mkdir -p /srv/salt-overlay/salt/kube-proxy
    (umask 077;
        cat > "${kube_proxy_kubeconfig_file}" <<EOF
apiVersion: v1
kind: Config
users:
- name: kube-proxy
  user:
    token: ${KUBE_PROXY_TOKEN}
clusters:
- name: local
  cluster:
    certificate-authority-data: ${CA_CERT}
contexts:
- context:
    cluster: local
    user: kube-proxy
  name: service-account-context
current-context: service-account-context
EOF
)
  fi
}

function try-download-release() {
  # TODO(zmerlynn): Now we REALLy have no excuse not to do the reboot
  # optimization.

  # TODO(zmerlynn): This may not be set yet by everyone (GKE).
  if [[ -z "${SERVER_BINARY_TAR_HASH:-}" ]]; then
    echo "Downloading binary release sha1 (not found in env)"
    download-or-bust "${SERVER_BINARY_TAR_URL}.sha1"
    SERVER_BINARY_TAR_HASH=$(cat "${SERVER_BINARY_TAR_URL##*/}.sha1")
  fi

  echo "Downloading binary release tar (${SERVER_BINARY_TAR_URL})"
  download-or-bust "${SERVER_BINARY_TAR_URL}"

  validate-hash "${SERVER_BINARY_TAR_URL##*/}" "${SERVER_BINARY_TAR_HASH}"
  echo "Validated ${SERVER_BINARY_TAR_URL} SHA1 = ${SERVER_BINARY_TAR_HASH}"

  # TODO(zmerlynn): This may not be set yet by everyone (GKE).
  if [[ -z "${SALT_TAR_HASH:-}" ]]; then
    echo "Downloading Salt tar sha1 (not found in env)"
    download-or-bust "${SALT_TAR_URL}.sha1"
    SALT_TAR_HASH=$(cat "${SALT_TAR_URL##*/}.sha1")
  fi

  echo "Downloading Salt tar ($SALT_TAR_URL)"
  download-or-bust "$SALT_TAR_URL"

  validate-hash "${SALT_TAR_URL##*/}" "${SALT_TAR_HASH}"
  echo "Validated ${SALT_TAR_URL} SHA1 = ${SALT_TAR_HASH}"

  echo "Unpacking Salt tree and checking integrity of binary release tar"
  rm -rf kubernetes
  tar xzf "${SALT_TAR_URL##*/}" && tar tzf "${SERVER_BINARY_TAR_URL##*/}" > /dev/null
}

function download-release() {
  # In case of failure checking integrity of release, retry.
  until try-download-release; do
    sleep 15
    echo "Couldn't download release. Retrying..."
  done

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
  if ! [[ -z "${PROJECT_ID:-}" ]] && ! [[ -z "${TOKEN_URL:-}" ]] && ! [[ -z "${TOKEN_BODY:-}" ]] && ! [[ -z "${NODE_NETWORK:-}" ]] ; then
    cat <<EOF >/etc/gce.conf
[global]
token-url = ${TOKEN_URL}
token-body = ${TOKEN_BODY}
project-id = ${PROJECT_ID}
network-name = ${NODE_NETWORK}
EOF
    EXTERNAL_IP=$(curl --fail --silent -H 'Metadata-Flavor: Google' "http://metadata/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip")
    cat <<EOF >>/etc/salt/minion.d/grains.conf
  cloud_config: /etc/gce.conf
  advertise_address: '${EXTERNAL_IP}'
  proxy_ssh_user: '${PROXY_SSH_USER}'
EOF
  fi

  # If the kubelet on the master is enabled, give it the same CIDR range
  # as a generic node.
  if [[ ! -z "${KUBELET_APISERVER:-}" ]] && [[ ! -z "${KUBELET_CERT:-}" ]] && [[ ! -z "${KUBELET_KEY:-}" ]]; then
    cat <<EOF >>/etc/salt/minion.d/grains.conf
  kubelet_api_servers: '${KUBELET_APISERVER}'
  cbr-cidr: 10.123.45.0/30
EOF
  else
    # If the kubelet is running disconnected from a master, give it a fixed
    # CIDR range.
    cat <<EOF >>/etc/salt/minion.d/grains.conf
  cbr-cidr: ${MASTER_IP_RANGE}
EOF
  fi
  if [[ ! -z "${RUNTIME_CONFIG:-}" ]]; then
    cat <<EOF >>/etc/salt/minion.d/grains.conf
  runtime_config: '$(echo "$RUNTIME_CONFIG" | sed -e "s/'/''/g")'
EOF
  fi
}

function salt-node-role() {
  cat <<EOF >/etc/salt/minion.d/grains.conf
grains:
  roles:
    - kubernetes-pool
  cbr-cidr: 10.123.45.0/30
  cloud: gce
  api_servers: '${KUBERNETES_MASTER_NAME}'
EOF
}

function salt-docker-opts() {
  DOCKER_OPTS=""

  if [[ -n "${EXTRA_DOCKER_OPTS-}" ]]; then
    DOCKER_OPTS="${EXTRA_DOCKER_OPTS}"
  fi

  if [[ -n "{DOCKER_OPTS}" ]]; then
    cat <<EOF >>/etc/salt/minion.d/grains.conf
  docker_opts: '$(echo "$DOCKER_OPTS" | sed -e "s/'/''/g")'
EOF
  fi
}

function configure-salt() {
  fix-apt-sources
  mkdir -p /etc/salt/minion.d
  salt-run-local
  if [[ "${KUBERNETES_MASTER}" == "true" ]]; then
    salt-master-role
    if [ -n "${KUBE_APISERVER_REQUEST_TIMEOUT:-}"  ]; then
        salt-apiserver-timeout-grain $KUBE_APISERVER_REQUEST_TIMEOUT
    fi
  else
    salt-node-role
    salt-docker-opts
  fi
  install-salt
  stop-salt-minion
}

function run-salt() {
  echo "== Calling Salt =="
  salt-call --local state.highstate || true
}

####################################################################################

if [[ -z "${is_push}" ]]; then
  echo "== kube-up node config starting =="
  set-broken-motd
  ensure-basic-networking
  ensure-install-dir
  set-kube-env
  [[ "${KUBERNETES_MASTER}" == "true" ]] && mount-master-pd
  create-salt-pillar
  if [[ "${KUBERNETES_MASTER}" == "true" ]]; then
    create-salt-master-auth
    create-salt-master-kubelet-auth
  else
    create-salt-kubelet-auth
    create-salt-kubeproxy-auth
  fi
  download-release
  configure-salt
  remove-docker-artifacts
  run-salt
  set-good-motd
  echo "== kube-up node config done =="
else
  echo "== kube-push node config starting =="
  ensure-basic-networking
  ensure-install-dir
  set-kube-env
  create-salt-pillar
  download-release
  run-salt
  echo "== kube-push node config done =="
fi
