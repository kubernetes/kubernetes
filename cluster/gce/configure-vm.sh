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

# If we have any arguments at all, this is a push and not just setup.
is_push=$@

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

function create-node-pki {
  echo "Creating node pki files"

  local -r pki_dir="/etc/kubernetes/pki"
  mkdir -p "${pki_dir}"

  if [[ -z "${CA_CERT_BUNDLE:-}" ]]; then
    CA_CERT_BUNDLE="${CA_CERT}"
  fi

  CA_CERT_BUNDLE_PATH="${pki_dir}/ca-certificates.crt"
  echo "${CA_CERT_BUNDLE}" | base64 --decode > "${CA_CERT_BUNDLE_PATH}"

  if [[ ! -z "${KUBELET_CERT:-}" && ! -z "${KUBELET_KEY:-}" ]]; then
    KUBELET_CERT_PATH="${pki_dir}/kubelet.crt"
    echo "${KUBELET_CERT}" | base64 --decode > "${KUBELET_CERT_PATH}"

    KUBELET_KEY_PATH="${pki_dir}/kubelet.key"
    echo "${KUBELET_KEY}" | base64 --decode > "${KUBELET_KEY_PATH}"
  fi
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

function config-ip-firewall {
  echo "Configuring IP firewall rules"

  if [[ "${ENABLE_METADATA_CONCEALMENT:-}" == "true" ]]; then
    echo "Add rule for metadata concealment"
    iptables -w -t nat -I PREROUTING -p tcp -d 169.254.169.254 --dport 80 -m comment --comment "metadata-concealment: bridge traffic to metadata server goes to metadata proxy" -j DNAT --to-destination 127.0.0.1:988
  fi
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
enable_metadata_proxy: '$(echo "$ENABLE_METADATA_CONCEALMENT" | sed -e "s/'/''/g")'
enable_metrics_server: '$(echo "$ENABLE_METRICS_SERVER" | sed -e "s/'/''/g")'
enable_pod_security_policy: '$(echo "$ENABLE_POD_SECURITY_POLICY" | sed -e "s/'/''/g")'
enable_rescheduler: '$(echo "$ENABLE_RESCHEDULER" | sed -e "s/'/''/g")'
logging_destination: '$(echo "$LOGGING_DESTINATION" | sed -e "s/'/''/g")'
elasticsearch_replicas: '$(echo "$ELASTICSEARCH_LOGGING_REPLICAS" | sed -e "s/'/''/g")'
enable_cluster_dns: '$(echo "$ENABLE_CLUSTER_DNS" | sed -e "s/'/''/g")'
cluster_dns_core_dns: '$(echo "$CLUSTER_DNS_CORE_DNS" | sed -e "s/'/''/g")'
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
initial_etcd_cluster_state: '$(echo "${INITIAL_ETCD_CLUSTER_STATE:-}" | sed -e "s/'/''/g")'
ca_cert_bundle_path: '$(echo "${CA_CERT_BUNDLE_PATH:-}" | sed -e "s/'/''/g")'
hostname: '$(echo "${ETCD_HOSTNAME:-$(hostname -s)}" | sed -e "s/'/''/g")'
enable_pod_priority: '$(echo "${ENABLE_POD_PRIORITY:-}" | sed -e "s/'/''/g")'
enable_default_storage_class: '$(echo "$ENABLE_DEFAULT_STORAGE_CLASS" | sed -e "s/'/''/g")'
kube_proxy_daemonset: '$(echo "$KUBE_PROXY_DAEMONSET" | sed -e "s/'/''/g")'
EOF
    if [ -n "${STORAGE_BACKEND:-}" ]; then
      cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
storage_backend: '$(echo "$STORAGE_BACKEND" | sed -e "s/'/''/g")'
EOF
    fi
    if [ -n "${STORAGE_MEDIA_TYPE:-}" ]; then
      cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
storage_media_type: '$(echo "$STORAGE_MEDIA_TYPE" | sed -e "s/'/''/g")'
EOF
    fi
    if [ -n "${KUBE_APISERVER_REQUEST_TIMEOUT_SEC:-}" ]; then
      cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
kube_apiserver_request_timeout_sec: '$(echo "$KUBE_APISERVER_REQUEST_TIMEOUT_SEC" | sed -e "s/'/''/g")'
EOF
    fi
    if [ -n "${ETCD_LIVENESS_PROBE_INITIAL_DELAY_SEC:-}" ]; then
      cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
etcd_liveness_probe_initial_delay: '$(echo "$ETCD_LIVENESS_PROBE_INITIAL_DELAY_SEC" | sed -e "s/'/''/g")'
EOF
    fi
    if [ -n "${KUBE_APISERVER_LIVENESS_PROBE_INITIAL_DELAY_SEC:-}" ]; then
      cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
kube_apiserver_liveness_probe_initial_delay: '$(echo "$KUBE_APISERVER_LIVENESS_PROBE_INITIAL_DELAY_SEC" | sed -e "s/'/''/g")'
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
    if [ -n "${ETCD_DOCKER_REPOSITORY:-}" ]; then
      cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
etcd_docker_repository: '$(echo "$ETCD_DOCKER_REPOSITORY" | sed -e "s/'/''/g")'
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
    if [ -n "${NON_MASTER_NODE_LABELS:-}" ]; then
      cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
non_master_node_labels: '$(echo "${NON_MASTER_NODE_LABELS}" | sed -e "s/'/''/g")'
EOF
    fi
    if [ -n "${NODE_TAINTS:-}" ]; then
      cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
node_taints: '$(echo "${NODE_TAINTS}" | sed -e "s/'/''/g")'
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
autoscaler_expander_config: '$(echo "${AUTOSCALER_EXPANDER_CONFIG}" | sed -e "s/'/''/g")'
EOF
    fi
    if [ -n "${SCHEDULING_ALGORITHM_PROVIDER:-}" ]; then
      cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
scheduling_algorithm_provider: '$(echo "${SCHEDULING_ALGORITHM_PROVIDER}" | sed -e "s/'/''/g")'
EOF
    fi
    if [ -n "${ENABLE_IP_ALIASES:-}" ]; then
      cat <<EOF >>/srv/salt-overlay/pillar/cluster-params.sls
enable_ip_aliases: '$(echo "$ENABLE_IP_ALIASES" | sed -e "s/'/''/g")'
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

# This should happen both on cluster initialization and node upgrades.
#
#  - Uses KUBELET_CA_CERT (falling back to CA_CERT), KUBELET_CERT, and
#    KUBELET_KEY to generate a kubeconfig file for the kubelet to securely
#    connect to the apiserver.

function create-salt-kubelet-auth() {
  local -r kubelet_kubeconfig_file="/srv/salt-overlay/salt/kubelet/bootstrap-kubeconfig"
  if [ ! -e "${kubelet_kubeconfig_file}" ]; then
    mkdir -p /srv/salt-overlay/salt/kubelet
    (umask 077;
      cat > "${kubelet_kubeconfig_file}" <<EOF
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
    server: https://${KUBERNETES_MASTER_NAME}
    certificate-authority: ${CA_CERT_BUNDLE_PATH}
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
#  - When run as static pods, use the CA_CERT and KUBE_PROXY_TOKEN to generate a
#    kubeconfig file for the kube-proxy to securely connect to the apiserver.
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
    certificate-authority-data: ${CA_CERT_BUNDLE}
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

function salt-node-role() {
  local -r kubelet_bootstrap_kubeconfig="/srv/salt-overlay/salt/kubelet/bootstrap-kubeconfig"
  local -r kubelet_kubeconfig="/srv/salt-overlay/salt/kubelet/kubeconfig"
  cat <<EOF >/etc/salt/minion.d/grains.conf
grains:
  roles:
    - kubernetes-pool
  cloud: gce
  api_servers: '${KUBERNETES_MASTER_NAME}'
  kubelet_bootstrap_kubeconfig: /var/lib/kubelet/bootstrap-kubeconfig
  kubelet_kubeconfig: /var/lib/kubelet/kubeconfig
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
  salt-node-role
  node-docker-opts
  salt-grains
  install-salt
  stop-salt-minion
}

function run-salt() {
  echo "== Calling Salt =="
  local rc=0
  for i in {0..6}; do
    salt-call --retcode-passthrough --local state.highstate && rc=0 || rc=$?
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

if [[ "${KUBERNETES_MASTER:-}" == "true" ]]; then
  echo "Support for debian master has been removed"
  exit 1
fi

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
  create-node-pki
  create-salt-pillar
  create-salt-kubelet-auth
  if [[ "${KUBE_PROXY_DAEMONSET:-}" != "true" ]]; then
    create-salt-kubeproxy-auth
  fi
  download-release
  configure-salt
  remove-docker-artifacts
  config-ip-firewall
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
