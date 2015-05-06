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

  # Infer master status from presence in node pool
  if [[ $(hostname) = ${NODE_INSTANCE_PREFIX}* ]]; then
    KUBERNETES_MASTER="false"
  else
    KUBERNETES_MASTER="true"
  fi

  if [[ "${KUBERNETES_MASTER}" != "true" ]] && [[ -z "${MINION_IP_RANGE:-}" ]]; then
    # This block of code should go away once the master can allocate CIDRs
    until MINION_IP_RANGE=$(curl-metadata node-ip-range); do
      echo 'Waiting for metadata MINION_IP_RANGE...'
      sleep 3
    done
  fi
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
  until curl --ipv4 -Lo "$file" --connect-timeout 20 --retry 6 --retry-delay 10 "$1"; do
    echo "Failed to download file ($1). Retrying."
  done
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

  ln -s -f /mnt/master-pd/var/etcd /var/etcd
  ln -s -f /mnt/master-pd/srv/kubernetes /srv/kubernetes
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
cluster_class_b: '$(echo "$KUBE_GCE_CLUSTER_CLASS_B" | sed -e "s/'/''/g")'
allocate_node_cidrs: '$(echo "$ALLOCATE_NODE_CIDRS" | sed -e "s/'/''/g")'
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
admission_control: '$(echo "$ADMISSION_CONTROL" | sed -e "s/'/''/g")'
EOF
}

# This should only happen on cluster initialization. Uses
# KUBE_PASSWORD and KUBE_USER to generate basic_auth.csv. Uses
# KUBE_BEARER_TOKEN, KUBELET_TOKEN, and KUBE_PROXY_TOKEN to generate
# known_tokens.csv (KNOWN_TOKENS_FILE). After the first boot and
# on upgrade, this file exists on the master-pd and should never
# be touched again (except perhaps an additional service account,
# see NB below.)
function create-salt-auth() {
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

    mkdir -p /srv/salt-overlay/salt/kubelet
    kubelet_auth_file="/srv/salt-overlay/salt/kubelet/kubernetes_auth"
    (umask 077;
      echo "{\"BearerToken\": \"${KUBELET_TOKEN}\", \"Insecure\": true }" > "${kubelet_auth_file}")

    mkdir -p /srv/salt-overlay/salt/kube-proxy
    kube_proxy_kubeconfig_file="/srv/salt-overlay/salt/kube-proxy/kubeconfig"
    # Make a kubeconfig file with the token.
    # TODO(etune): put apiserver certs into secret too, and reference from authfile,
    # so that "Insecure" is not needed.
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
     insecure-skip-tls-verify: true
contexts:
- context:
    cluster: local
    user: kube-proxy
  name: service-account-context
current-context: service-account-context
EOF
)

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

function download-release() {
  # TODO(zmerlynn): We should optimize for the reboot case here, but
  # unlike the .debs, we don't have version information in the
  # filenames here, nor do the URLs even provide useful information in
  # the dev environment case (because they're just a project
  # bucket). We should probably push a hash into the kube-env, and
  # store it when we download, and then when it's different infer that
  # a push occurred (otherwise it's a simple reboot).

  echo "Downloading binary release tar ($SERVER_BINARY_TAR_URL)"
  download-or-bust "$SERVER_BINARY_TAR_URL"

  echo "Downloading Salt tar ($SALT_TAR_URL)"
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
  cbr-cidr: ${MASTER_IP_RANGE}
  cloud: gce
EOF
  if ! [[ -z "${PROJECT_ID:-}" ]] && ! [[ -z "${TOKEN_URL:-}" ]]; then
    cat <<EOF >/etc/gce.conf
[global]
token-url = ${TOKEN_URL}
project-id = ${PROJECT_ID}
EOF
    cat <<EOF >>/etc/salt/minion.d/grains.conf
  cloud_config: /etc/gce.conf
EOF
  fi
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
  local kube_master_fqdn
  until kube_master_fqdn=$(getent hosts ${KUBERNETES_MASTER_NAME} | awk '{ print $2 }'); do
    echo 'Waiting for DNS resolution of ${KUBERNETES_MASTER_NAME}...'
    sleep 3
  done

  cat <<EOF >>/etc/salt/minion.d/grains.conf
  api_servers: '${kube_master_fqdn}'
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
  echo "== Calling Salt =="
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
  download-release
  run-salt
  echo "== kube-push node config done =="
fi
