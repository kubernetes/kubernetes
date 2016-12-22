#!/bin/bash

# Copyright 2015 The Kubernetes Authors.
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

# Note that this script is also used by AWS; we include it and then override
# functions with AWS equivalents.  Note `#+AWS_OVERRIDES_HERE` below.
# TODO(justinsb): Refactor into common script & GCE specific script?

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
  until getent hosts $(hostname -f || echo _error_) &>/dev/null; do
    echo 'Waiting for functional DNS (trying to resolve my own FQDN)...'
    sleep 3
  done
  until getent hosts $(hostname -i || echo _error_) &>/dev/null; do
    echo 'Waiting for functional DNS (trying to resolve my own IP)...'
    sleep 3
  done

  echo "Networking functional on $(hostname) ($(hostname -i))"
}

# A hookpoint for installing any needed packages
ensure-packages() {
  :
}

# A hookpoint for setting up local devices
ensure-local-disks() {
 for ssd in /dev/disk/by-id/google-local-ssd-*; do
    if [ -e "$ssd" ]; then
      ssdnum=`echo $ssd | sed -e 's/\/dev\/disk\/by-id\/google-local-ssd-\([0-9]*\)/\1/'`
      echo "Formatting and mounting local SSD $ssd to /mnt/disks/ssd$ssdnum"
      mkdir -p /mnt/disks/ssd$ssdnum
      /usr/share/google/safe_format_and_mount -m "mkfs.ext4 -F" "${ssd}" /mnt/disks/ssd$ssdnum &>/var/log/local-ssd-$ssdnum-mount.log || \
      { echo "Local SSD $ssdnum mount failed, review /var/log/local-ssd-$ssdnum-mount.log"; return 1; }
    else
      echo "No local SSD disks found."
    fi
  done
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
  echo -e '\nBroken (or in progress) Kubernetes node setup! Suggested first step:\n  tail /var/log/startupscript.log\n' > /etc/motd
}

function reset-motd() {
  # kubelet is installed both on the master and nodes, and the version is easy to parse (unlike kubectl)
  local -r version="$(/usr/local/bin/kubelet --version=true | cut -f2 -d " ")"
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
  /usr/local/share/doc/kubernetes/kubernetes-src.tar.gz
Or you can download it at:
  https://storage.googleapis.com/kubernetes-release/release/${version}/kubernetes-src.tar.gz

It is based on the Kubernetes source at:
  https://github.com/kubernetes/kubernetes/tree/${gitref}
${devel}
For Kubernetes copyright and licensing information, see:
  /usr/local/share/doc/kubernetes/LICENSES

EOF
}

function curl-metadata() {
  curl --fail --retry 5 --silent -H 'Metadata-Flavor: Google' "http://metadata/computeMetadata/v1/instance/attributes/${1}"
}

function set-kube-env() {
  local kube_env_yaml="${INSTALL_DIR}/kube_env.yaml"

  until curl-metadata kube-env > "${kube_env_yaml}"; do
    echo 'Waiting for kube-env...'
    sleep 3
  done

  # kube-env has all the environment variables we care about, in a flat yaml format
  eval "$(python -c '
import pipes,sys,yaml

for k,v in yaml.load(sys.stdin).iteritems():
  print("""readonly {var}={value}""".format(var = k, value = pipes.quote(str(v))))
  print("""export {var}""".format(var = k))
  ' < """${kube_env_yaml}""")"
}

function remove-docker-artifacts() {
  echo "== Deleting docker0 =="
  apt-get-install bridge-utils

  # Remove docker artifacts on minion nodes, if present
  iptables -t nat -F || true
  ifconfig docker0 down || true
  brctl delbr docker0 || true
  echo "== Finished deleting docker0 =="
}

# Retry a download until we get it. Takes a hash and a set of URLs.
#
# $1 is the sha1 of the URL. Can be "" if the sha1 is unknown.
# $2+ are the URLs to download.
download-or-bust() {
  local -r hash="$1"
  shift 1

  urls=( $* )
  while true; do
    for url in "${urls[@]}"; do
      local file="${url##*/}"
      rm -f "${file}"
      if ! curl -f --ipv4 -Lo "${file}" --connect-timeout 20 --max-time 300 --retry 6 --retry-delay 10 "${url}"; then
        echo "== Failed to download ${url}. Retrying. =="
      elif [[ -n "${hash}" ]] && ! validate-hash "${file}" "${hash}"; then
        echo "== Hash validation of ${url} failed. Retrying. =="
      else
        if [[ -n "${hash}" ]]; then
          echo "== Downloaded ${url} (SHA1 = ${hash}) =="
        else
          echo "== Downloaded ${url} =="
        fi
        return
      fi
    done
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

apt-get-install() {
  local -r packages=( $@ )
  installed=true
  for package in "${packages[@]}"; do
    if ! dpkg -s "${package}" &>/dev/null; then
      installed=false
      break
    fi
  done
  if [[ "${installed}" == "true" ]]; then
    echo "== ${packages[@]} already installed, skipped apt-get install ${packages[@]} =="
    return
  fi

  apt-get-update

  # Forcibly install packages (options borrowed from Salt logs).
  until apt-get -q -y -o DPkg::Options::=--force-confold -o DPkg::Options::=--force-confdef install $@; do
    echo "== install of packages $@ failed, retrying =="
    sleep 5
  done
}

apt-get-update() {
  echo "== Refreshing package database =="
  until apt-get update; do
    echo "== apt-get update failed, retrying =="
    sleep 5
  done
}

# Restart any services that need restarting due to a library upgrade
# Uses needrestart
restart-updated-services() {
  # We default to restarting services, because this is only done as part of an update
  if [[ "${AUTO_RESTART_SERVICES:-true}" != "true" ]]; then
    echo "Auto restart of services prevented by AUTO_RESTART_SERVICES=${AUTO_RESTART_SERVICES}"
    return
  fi
  echo "Restarting services with updated libraries (needrestart -r a)"
  # The pipes make sure that needrestart doesn't think it is running with a TTY
  # Debian bug #803249; fixed but not necessarily in package repos yet
  echo "" | needrestart -r a 2>&1 | tee /dev/null
}

# Reboot the machine if /var/run/reboot-required exists
reboot-if-required() {
  if [[ ! -e "/var/run/reboot-required" ]]; then
    return
  fi

  echo "Reboot is required (/var/run/reboot-required detected)"
  if [[ -e "/var/run/reboot-required.pkgs" ]]; then
    echo "Packages that triggered reboot:"
    cat /var/run/reboot-required.pkgs
  fi

  # We default to rebooting the machine because this is only done as part of an update
  if [[ "${AUTO_REBOOT:-true}" != "true" ]]; then
    echo "Reboot prevented by AUTO_REBOOT=${AUTO_REBOOT}"
    return
  fi

  rm -f /var/run/reboot-required
  rm -f /var/run/reboot-required.pkgs
  echo "Triggering reboot"
  init 6
}

# Install upgrades using unattended-upgrades, then reboot or restart services
auto-upgrade() {
  # We default to not installing upgrades
  if [[ "${AUTO_UPGRADE:-false}" != "true" ]]; then
    echo "AUTO_UPGRADE not set to true; won't auto-upgrade"
    return
  fi
  apt-get-install unattended-upgrades needrestart
  unattended-upgrade --debug
  reboot-if-required # We may reboot the machine right here
  restart-updated-services
}

#
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
    sleep 5
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
      download-or-bust "" "${URL_BASE}/${deb}"
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
    sleep 5
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

# Finds the master PD device; returns it in MASTER_PD_DEVICE
find-master-pd() {
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
#
# This function can be reused to mount an existing PD because all of its
# operations modifying the disk are idempotent -- safe_format_and_mount only
# formats an unformatted disk, and mkdir -p will leave a directory be if it
# already exists.
mount-master-pd() {
  find-master-pd
  if [[ -z "${MASTER_PD_DEVICE:-}" ]]; then
    return
  fi

  # Format and mount the disk, create directories on it for all of the master's
  # persistent data, and link them to where they're used.
  echo "Mounting master-pd"
  mkdir -p /mnt/master-pd
  /usr/share/google/safe_format_and_mount -m "mkfs.ext4 -F" "${MASTER_PD_DEVICE}" /mnt/master-pd &>/var/log/master-pd-mount.log || \
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
node_tags: '$(echo "$NODE_TAGS" | sed -e "s/'/''/g")'
node_instance_prefix: '$(echo "$NODE_INSTANCE_PREFIX" | sed -e "s/'/''/g")'
cluster_cidr: '$(echo "$CLUSTER_IP_RANGE" | sed -e "s/'/''/g")'
allocate_node_cidrs: '$(echo "$ALLOCATE_NODE_CIDRS" | sed -e "s/'/''/g")'
non_masquerade_cidr: '$(echo "$NON_MASQUERADE_CIDR" | sed -e "s/'/''/g")'
service_cluster_ip_range: '$(echo "$SERVICE_CLUSTER_IP_RANGE" | sed -e "s/'/''/g")'
enable_cluster_monitoring: '$(echo "$ENABLE_CLUSTER_MONITORING" | sed -e "s/'/''/g")'
enable_cluster_logging: '$(echo "$ENABLE_CLUSTER_LOGGING" | sed -e "s/'/''/g")'
enable_cluster_ui: '$(echo "$ENABLE_CLUSTER_UI" | sed -e "s/'/''/g")'
enable_node_problem_detector: '$(echo "$ENABLE_NODE_PROBLEM_DETECTOR" | sed -e "s/'/''/g")'
enable_l7_loadbalancing: '$(echo "$ENABLE_L7_LOADBALANCING" | sed -e "s/'/''/g")'
enable_node_logging: '$(echo "$ENABLE_NODE_LOGGING" | sed -e "s/'/''/g")'
enable_rescheduler: '$(echo "$ENABLE_RESCHEDULER" | sed -e "s/'/''/g")'
logging_destination: '$(echo "$LOGGING_DESTINATION" | sed -e "s/'/''/g")'
elasticsearch_replicas: '$(echo "$ELASTICSEARCH_LOGGING_REPLICAS" | sed -e "s/'/''/g")'
enable_cluster_dns: '$(echo "$ENABLE_CLUSTER_DNS" | sed -e "s/'/''/g")'
enable_cluster_registry: '$(echo "$ENABLE_CLUSTER_REGISTRY" | sed -e "s/'/''/g")'
dns_server: '$(echo "$DNS_SERVER_IP" | sed -e "s/'/''/g")'
dns_domain: '$(echo "$DNS_DOMAIN" | sed -e "s/'/''/g")'
enable_dns_horizontal_autoscaler: '$(echo "$ENABLE_DNS_HORIZONTAL_AUTOSCALER" | sed -e "s/'/''/g")'
admission_control: '$(echo "$ADMISSION_CONTROL" | sed -e "s/'/''/g")'
network_provider: '$(echo "$NETWORK_PROVIDER" | sed -e "s/'/''/g")'
prepull_e2e_images: '$(echo "$PREPULL_E2E_IMAGES" | sed -e "s/'/''/g")'
hairpin_mode: '$(echo "$HAIRPIN_MODE" | sed -e "s/'/''/g")'
softlockup_panic: '$(echo "$SOFTLOCKUP_PANIC" | sed -e "s/'/''/g")'
opencontrail_tag: '$(echo "$OPENCONTRAIL_TAG" | sed -e "s/'/''/g")'
opencontrail_kubernetes_tag: '$(echo "$OPENCONTRAIL_KUBERNETES_TAG")'
opencontrail_public_subnet: '$(echo "$OPENCONTRAIL_PUBLIC_SUBNET")'
network_policy_provider: '$(echo "$NETWORK_POLICY_PROVIDER" | sed -e "s/'/''/g")'
enable_manifest_url: '$(echo "${ENABLE_MANIFEST_URL:-}" | sed -e "s/'/''/g")'
manifest_url: '$(echo "${MANIFEST_URL:-}" | sed -e "s/'/''/g")'
manifest_url_header: '$(echo "${MANIFEST_URL_HEADER:-}" | sed -e "s/'/''/g")'
num_nodes: $(echo "${NUM_NODES:-}" | sed -e "s/'/''/g")
e2e_storage_test_environment: '$(echo "$E2E_STORAGE_TEST_ENVIRONMENT" | sed -e "s/'/''/g")'
kube_uid: '$(echo "${KUBE_UID}" | sed -e "s/'/''/g")'
initial_etcd_cluster: '$(echo "${INITIAL_ETCD_CLUSTER:-}" | sed -e "s/'/''/g")'

hostname: $(hostname -s)
enable_default_storage_class: '$(echo "$ENABLE_DEFAULT_STORAGE_CLASS" | sed -e "s/'/''/g")'
EOF
    if [ -n "${STORAGE_BACKEND:-}" ]; then
      cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
storage_backend: '$(echo "$STORAGE_BACKEND" | sed -e "s/'/''/g")'
EOF
    fi
    if [ -n "${ADMISSION_CONTROL:-}" ] && [ ${ADMISSION_CONTROL} == *"ImagePolicyWebhook"* ]; then
      cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
admission-control-config-file: /etc/admission_controller.config
EOF
    fi
    if [ -n "${KUBELET_PORT:-}" ]; then
      cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
kubelet_port: '$(echo "$KUBELET_PORT" | sed -e "s/'/''/g")'
EOF
    fi
    if [ -n "${ETCD_IMAGE:-}" ]; then
      cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
etcd_docker_tag: '$(echo "$ETCD_IMAGE" | sed -e "s/'/''/g")'
EOF
    fi
    if [ -n "${ETCD_VERSION:-}" ]; then
      cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
etcd_version: '$(echo "$ETCD_VERSION" | sed -e "s/'/''/g")'
EOF
    fi
    if [[ -n "${ETCD_CA_KEY:-}" && -n "${ETCD_CA_CERT:-}" && -n "${ETCD_PEER_KEY:-}" && -n "${ETCD_PEER_CERT:-}" ]]; then
      cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
etcd_over_ssl: 'true'
EOF
    else
      cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
etcd_over_ssl: 'false'
EOF
    fi
    if [ -n "${ETCD_QUORUM_READ:-}" ]; then
      cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
etcd_quorum_read: '$(echo "${ETCD_QUORUM_READ}" | sed -e "s/'/''/g")'
EOF
    fi
    # Configuration changes for test clusters
    if [ -n "${APISERVER_TEST_ARGS:-}" ]; then
      cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
apiserver_test_args: '$(echo "$APISERVER_TEST_ARGS" | sed -e "s/'/''/g")'
EOF
    fi
    if [ -n "${API_SERVER_TEST_LOG_LEVEL:-}" ]; then
      cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
api_server_test_log_level: '$(echo "$API_SERVER_TEST_LOG_LEVEL" | sed -e "s/'/''/g")'
EOF
    fi
    if [ -n "${KUBELET_TEST_ARGS:-}" ]; then
      cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
kubelet_test_args: '$(echo "$KUBELET_TEST_ARGS" | sed -e "s/'/''/g")'
EOF
    fi
    if [ -n "${KUBELET_TEST_LOG_LEVEL:-}" ]; then
      cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
kubelet_test_log_level: '$(echo "$KUBELET_TEST_LOG_LEVEL" | sed -e "s/'/''/g")'
EOF
    fi
    if [ -n "${DOCKER_TEST_LOG_LEVEL:-}" ]; then
      cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
docker_test_log_level: '$(echo "$DOCKER_TEST_LOG_LEVEL" | sed -e "s/'/''/g")'
EOF
    fi
    if [ -n "${CONTROLLER_MANAGER_TEST_ARGS:-}" ]; then
      cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
controller_manager_test_args: '$(echo "$CONTROLLER_MANAGER_TEST_ARGS" | sed -e "s/'/''/g")'
EOF
    fi
    if [ -n "${CONTROLLER_MANAGER_TEST_LOG_LEVEL:-}" ]; then
      cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
controller_manager_test_log_level: '$(echo "$CONTROLLER_MANAGER_TEST_LOG_LEVEL" | sed -e "s/'/''/g")'
EOF
    fi
    if [ -n "${SCHEDULER_TEST_ARGS:-}" ]; then
      cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
scheduler_test_args: '$(echo "$SCHEDULER_TEST_ARGS" | sed -e "s/'/''/g")'
EOF
    fi
    if [ -n "${SCHEDULER_TEST_LOG_LEVEL:-}" ]; then
      cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
scheduler_test_log_level: '$(echo "$SCHEDULER_TEST_LOG_LEVEL" | sed -e "s/'/''/g")'
EOF
    fi
    if [ -n "${KUBEPROXY_TEST_ARGS:-}" ]; then
      cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
kubeproxy_test_args: '$(echo "$KUBEPROXY_TEST_ARGS" | sed -e "s/'/''/g")'
EOF
    fi
    if [ -n "${KUBEPROXY_TEST_LOG_LEVEL:-}" ]; then
      cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
kubeproxy_test_log_level: '$(echo "$KUBEPROXY_TEST_LOG_LEVEL" | sed -e "s/'/''/g")'
EOF
    fi
    # TODO: Replace this  with a persistent volume (and create it).
    if [[ "${ENABLE_CLUSTER_REGISTRY}" == true && -n "${CLUSTER_REGISTRY_DISK}" ]]; then
      cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
cluster_registry_disk_type: gce
cluster_registry_disk_size: $(echo $(convert-bytes-gce-kube ${CLUSTER_REGISTRY_DISK_SIZE}) | sed -e "s/'/''/g")
cluster_registry_disk_name: $(echo ${CLUSTER_REGISTRY_DISK} | sed -e "s/'/''/g")
EOF
    fi
    if [ -n "${TERMINATED_POD_GC_THRESHOLD:-}" ]; then
      cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
terminated_pod_gc_threshold: '$(echo "${TERMINATED_POD_GC_THRESHOLD}" | sed -e "s/'/''/g")'
EOF
    fi
    if [ -n "${ENABLE_CUSTOM_METRICS:-}" ]; then
      cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
enable_custom_metrics: '$(echo "${ENABLE_CUSTOM_METRICS}" | sed -e "s/'/''/g")'
EOF
    fi
    if [ -n "${NODE_LABELS:-}" ]; then
      cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
node_labels: '$(echo "${NODE_LABELS}" | sed -e "s/'/''/g")'
EOF
    fi
    if [ -n "${EVICTION_HARD:-}" ]; then
      cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
eviction_hard: '$(echo "${EVICTION_HARD}" | sed -e "s/'/''/g")'
EOF
    fi
    if [[ "${ENABLE_CLUSTER_AUTOSCALER:-false}" == "true" ]]; then
      cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
enable_cluster_autoscaler: '$(echo "${ENABLE_CLUSTER_AUTOSCALER}" | sed -e "s/'/''/g")'
autoscaler_mig_config: '$(echo "${AUTOSCALER_MIG_CONFIG}" | sed -e "s/'/''/g")'
EOF
    fi
    if [[ "${FEDERATION:-}" == "true" ]]; then
      local federations_domain_map="${FEDERATIONS_DOMAIN_MAP:-}"
      if [[ -z "${federations_domain_map}" && -n "${FEDERATION_NAME:-}" && -n "${DNS_ZONE_NAME:-}" ]]; then
        federations_domain_map="${FEDERATION_NAME}=${DNS_ZONE_NAME}"
      fi
      if [[ -n "${federations_domain_map}" ]]; then
        cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
federations_domain_map: '$(echo "- --federations=${federations_domain_map}" | sed -e "s/'/''/g")'
EOF
      else
        cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
federations_domain_map: ''
EOF
      fi
    else
      cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
federations_domain_map: ''
EOF
    fi
    if [ -n "${SCHEDULING_ALGORITHM_PROVIDER:-}" ]; then
      cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
scheduling_algorithm_provider: '$(echo "${SCHEDULING_ALGORITHM_PROVIDER}" | sed -e "s/'/''/g")'
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
# After the first boot and on upgrade, these files exist on the master-pd
# and should never be touched again (except perhaps an additional service
# account, see NB below.)
function create-salt-master-auth() {
  if [[ ! -e /srv/kubernetes/ca.crt ]]; then
    if  [[ ! -z "${CA_CERT:-}" ]] && [[ ! -z "${MASTER_CERT:-}" ]] && [[ ! -z "${MASTER_KEY:-}" ]]; then
      mkdir -p /srv/kubernetes
      (umask 077;
        echo "${CA_CERT}" | base64 --decode > /srv/kubernetes/ca.crt;
        echo "${MASTER_CERT}" | base64 --decode > /srv/kubernetes/server.cert;
        echo "${MASTER_KEY}" | base64 --decode > /srv/kubernetes/server.key;
        # Kubecfg cert/key are optional and included for backwards compatibility.
        # TODO(roberthbailey): Remove these two lines once GKE no longer requires
        # fetching clients certs from the master VM.
        echo "${KUBECFG_CERT:-}" | base64 --decode > /srv/kubernetes/kubecfg.crt;
        echo "${KUBECFG_KEY:-}" | base64 --decode > /srv/kubernetes/kubecfg.key)
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
    server: https://kubernetes-master
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

function split-commas() {
  echo $1 | tr "," "\n"
}

function try-download-release() {
  # TODO(zmerlynn): Now we REALLy have no excuse not to do the reboot
  # optimization.

  local -r server_binary_tar_urls=( $(split-commas "${SERVER_BINARY_TAR_URL}") )
  local -r server_binary_tar="${server_binary_tar_urls[0]##*/}"
  if [[ -n "${SERVER_BINARY_TAR_HASH:-}" ]]; then
    local -r server_binary_tar_hash="${SERVER_BINARY_TAR_HASH}"
  else
    echo "Downloading binary release sha1 (not found in env)"
    download-or-bust "" "${server_binary_tar_urls[@]/.tar.gz/.tar.gz.sha1}"
    local -r server_binary_tar_hash=$(cat "${server_binary_tar}.sha1")
  fi

  echo "Downloading binary release tar (${server_binary_tar_urls[@]})"
  download-or-bust "${server_binary_tar_hash}" "${server_binary_tar_urls[@]}"

  local -r salt_tar_urls=( $(split-commas "${SALT_TAR_URL}") )
  local -r salt_tar="${salt_tar_urls[0]##*/}"
  if [[ -n "${SALT_TAR_HASH:-}" ]]; then
    local -r salt_tar_hash="${SALT_TAR_HASH}"
  else
    echo "Downloading Salt tar sha1 (not found in env)"
    download-or-bust "" "${salt_tar_urls[@]/.tar.gz/.tar.gz.sha1}"
    local -r salt_tar_hash=$(cat "${salt_tar}.sha1")
  fi

  echo "Downloading Salt tar (${salt_tar_urls[@]})"
  download-or-bust "${salt_tar_hash}" "${salt_tar_urls[@]}"

  echo "Unpacking Salt tree and checking integrity of binary release tar"
  rm -rf kubernetes
  tar xzf "${salt_tar}" && tar tzf "${server_binary_tar}" > /dev/null
}

function download-release() {
  # In case of failure checking integrity of release, retry.
  until try-download-release; do
    sleep 15
    echo "Couldn't download release. Retrying..."
  done

  echo "Running release install script"
  kubernetes/saltbase/install.sh "${SERVER_BINARY_TAR_URL##*/}"
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

  cat <<EOF >/etc/gce.conf
[global]
EOF
  CLOUD_CONFIG='' # Set to non-empty path if we are using the gce.conf file

  if ! [[ -z "${PROJECT_ID:-}" ]] && ! [[ -z "${TOKEN_URL:-}" ]] && ! [[ -z "${TOKEN_BODY:-}" ]] && ! [[ -z "${NODE_NETWORK:-}" ]] ; then
    cat <<EOF >>/etc/gce.conf
token-url = ${TOKEN_URL}
token-body = ${TOKEN_BODY}
project-id = ${PROJECT_ID}
network-name = ${NODE_NETWORK}
EOF
    CLOUD_CONFIG=/etc/gce.conf
    EXTERNAL_IP=$(curl --fail --silent -H 'Metadata-Flavor: Google' "http://metadata/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip")
    cat <<EOF >>/etc/salt/minion.d/grains.conf
  advertise_address: '${EXTERNAL_IP}'
  proxy_ssh_user: '${PROXY_SSH_USER}'
EOF
  fi

  if [[ -n "${NODE_INSTANCE_PREFIX:-}" ]]; then
    if [[ -n "${NODE_TAGS:-}" ]]; then
      local -r node_tags="${NODE_TAGS}"
    else
      local -r node_tags="${NODE_INSTANCE_PREFIX}"
    fi
    cat <<EOF >>/etc/gce.conf
node-tags = ${NODE_TAGS}
node-instance-prefix = ${NODE_INSTANCE_PREFIX}
EOF
    CLOUD_CONFIG=/etc/gce.conf
  fi

  if [[ -n "${MULTIZONE:-}" ]]; then
    cat <<EOF >>/etc/gce.conf
multizone = ${MULTIZONE}
EOF
    CLOUD_CONFIG=/etc/gce.conf
  fi

  if [[ -n "${CLOUD_CONFIG:-}" ]]; then
    cat <<EOF >>/etc/salt/minion.d/grains.conf
  cloud_config: ${CLOUD_CONFIG}
EOF
  else
    rm -f /etc/gce.conf
  fi

  if [[ -n "${GCP_AUTHN_URL:-}" ]]; then
    cat <<EOF >>/etc/salt/minion.d/grains.conf
  webhook_authentication_config: /etc/gcp_authn.config
EOF
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
    cat <<EOF >>/etc/salt/minion.d/grains.conf
  webhook_authorization_config: /etc/gcp_authz.config
EOF
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
    cat <<EOF >>/etc/salt/minion.d/grains.conf
  image_review_config: /etc/gcp_image_review.config
EOF
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
    cat <<EOF >>/etc/salt/minion.d/grains.conf
  image_review_webhook_config: /etc/admission_controller.config
EOF
    cat <<EOF >/etc/admission_controller.config
imagePolicy:
  kubeConfigFile: /etc/gcp_image_review.config
  allowTTL: 30
  denyTTL: 30
  retryBackoff: 500
  defaultAllow: true
EOF
  fi


  # If the kubelet on the master is enabled, give it the same CIDR range
  # as a generic node.
  if [[ ! -z "${KUBELET_APISERVER:-}" ]] && [[ ! -z "${KUBELET_CERT:-}" ]] && [[ ! -z "${KUBELET_KEY:-}" ]]; then
    cat <<EOF >>/etc/salt/minion.d/grains.conf
  kubelet_api_servers: '${KUBELET_APISERVER}'
EOF
  else
    # If the kubelet is running disconnected from a master, give it a fixed
    # CIDR range.
    cat <<EOF >>/etc/salt/minion.d/grains.conf
  cbr-cidr: ${MASTER_IP_RANGE}
EOF
  fi

  env-to-grains "runtime_config"
  env-to-grains "kube_user"
}

function salt-node-role() {
  cat <<EOF >/etc/salt/minion.d/grains.conf
grains:
  roles:
    - kubernetes-pool
  cloud: gce
  api_servers: '${KUBERNETES_MASTER_NAME}'
EOF
}

function env-to-grains {
  local key=$1
  local env_key=`echo $key | tr '[:lower:]' '[:upper:]'`
  local value=${!env_key:-}
  if [[ -n "${value}" ]]; then
    # Note this is yaml, so indentation matters
    cat <<EOF >>/etc/salt/minion.d/grains.conf
  ${key}: '$(echo "${value}" | sed -e "s/'/''/g")'
EOF
  fi
}

function node-docker-opts() {
  if [[ -n "${EXTRA_DOCKER_OPTS-}" ]]; then
    DOCKER_OPTS="${DOCKER_OPTS:-} ${EXTRA_DOCKER_OPTS}"
  fi

  # Decide whether to enable a docker registry mirror. This is taken from
  # the "kube-env" metadata value.
  if [[ -n "${DOCKER_REGISTRY_MIRROR_URL:-}" ]]; then
    echo "Enable docker registry mirror at: ${DOCKER_REGISTRY_MIRROR_URL}"
    DOCKER_OPTS="${DOCKER_OPTS:-} --registry-mirror=${DOCKER_REGISTRY_MIRROR_URL}"
  fi
}

function salt-grains() {
  env-to-grains "docker_opts"
  env-to-grains "docker_root"
  env-to-grains "kubelet_root"
  env-to-grains "feature_gates"
}

function configure-salt() {
  mkdir -p /etc/salt/minion.d
  salt-run-local
  if [[ "${KUBERNETES_MASTER}" == "true" ]]; then
    salt-master-role
    if [ -n "${KUBE_APISERVER_REQUEST_TIMEOUT:-}"  ]; then
        salt-apiserver-timeout-grain $KUBE_APISERVER_REQUEST_TIMEOUT
    fi
  else
    salt-node-role
    node-docker-opts
  fi
  salt-grains
  install-salt
  stop-salt-minion
}

function run-salt() {
  echo "== Calling Salt =="
  local rc=0
  for i in {0..6}; do
    salt-call --local state.highstate && rc=0 || rc=$?
    if [[ "${rc}" == 0 ]]; then
      return 0
    fi
  done
  echo "Salt failed to run repeatedly" >&2
  return "${rc}"
}

function run-user-script() {
  if curl-metadata k8s-user-startup-script > "${INSTALL_DIR}/k8s-user-script.sh"; then
    user_script=$(cat "${INSTALL_DIR}/k8s-user-script.sh")
  fi
  if [[ ! -z ${user_script:-} ]]; then
    chmod u+x "${INSTALL_DIR}/k8s-user-script.sh"
    echo "== running user startup script =="
    "${INSTALL_DIR}/k8s-user-script.sh"
  fi
}

function create-salt-master-etcd-auth {
  if [[ -n "${ETCD_CA_CERT:-}" && -n "${ETCD_PEER_KEY:-}" && -n "${ETCD_PEER_CERT:-}" ]]; then
    local -r auth_dir="/srv/kubernetes"
    echo "${ETCD_CA_CERT}" | base64 --decode | gunzip > "${auth_dir}/etcd-ca.crt"
    echo "${ETCD_PEER_KEY}" | base64 --decode > "${auth_dir}/etcd-peer.key"
    echo "${ETCD_PEER_CERT}" | base64 --decode | gunzip > "${auth_dir}/etcd-peer.crt"
  fi
}

# This script is re-used on AWS.  Some of the above functions will be replaced.
# The AWS kube-up script looks for this marker:
#+AWS_OVERRIDES_HERE

####################################################################################

if [[ -z "${is_push}" ]]; then
  echo "== kube-up node config starting =="
  set-broken-motd
  ensure-basic-networking
  fix-apt-sources
  ensure-install-dir
  ensure-packages
  set-kube-env
  auto-upgrade
  ensure-local-disks
  [[ "${KUBERNETES_MASTER}" == "true" ]] && mount-master-pd
  create-salt-pillar
  if [[ "${KUBERNETES_MASTER}" == "true" ]]; then
    create-salt-master-auth
    create-salt-master-etcd-auth
    create-salt-master-kubelet-auth
  else
    create-salt-kubelet-auth
    create-salt-kubeproxy-auth
  fi
  download-release
  configure-salt
  remove-docker-artifacts
  run-salt
  reset-motd

  run-user-script
  echo "== kube-up node config done =="
else
  echo "== kube-push node config starting =="
  ensure-basic-networking
  ensure-install-dir
  set-kube-env
  create-salt-pillar
  download-release
  reset-motd
  run-salt
  echo "== kube-push node config done =="
fi
