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

function config-ip-firewall {
  echo "Configuring IP firewall rules"
  # The GCI image has host firewall which drop most inbound/forwarded packets.
  # We need to add rules to accept all TCP/UDP/ICMP packets.
  if iptables -L INPUT | grep "Chain INPUT (policy DROP)" > /dev/null; then
    echo "Add rules to accept all inbound TCP/UDP/ICMP packets"
    iptables -A INPUT -w -p TCP -j ACCEPT
    iptables -A INPUT -w -p UDP -j ACCEPT
    iptables -A INPUT -w -p ICMP -j ACCEPT
  fi
  if iptables -L FORWARD | grep "Chain FORWARD (policy DROP)" > /dev/null; then
    echo "Add rules to accept all forwarded TCP/UDP/ICMP packets"
    iptables -A FORWARD -w -p TCP -j ACCEPT
    iptables -A FORWARD -w -p UDP -j ACCEPT
    iptables -A FORWARD -w -p ICMP -j ACCEPT
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

# Formats the given device ($1) if needed and mounts it at given mount point
# ($2).
function safe-format-and-mount() {
  device=$1
  mountpoint=$2

  # Format only if the disk is not already formatted.
  if ! tune2fs -l "${device}" ; then
    echo "Formatting '${device}'"
    mkfs.ext4 -F -E lazy_itable_init=0,lazy_journal_init=0,discard "${device}"
  fi

  mkdir -p "${mountpoint}"
  echo "Mounting '${device}' at '${mountpoint}'"
  mount -o discard,defaults "${device}" "${mountpoint}"
}

# Local ssds, if present, are mounted at /mnt/disks/ssdN.
function ensure-local-ssds() {
  for ssd in /dev/disk/by-id/google-local-ssd-*; do
    if [ -e "${ssd}" ]; then
      ssdnum=`echo ${ssd} | sed -e 's/\/dev\/disk\/by-id\/google-local-ssd-\([0-9]*\)/\1/'`
      ssdmount="/mnt/disks/ssd${ssdnum}/"
      mkdir -p ${ssdmount}
      safe-format-and-mount "${ssd}" ${ssdmount}
      echo "Mounted local SSD $ssd at ${ssdmount}"
      chmod a+w ${ssdmount}
    else
      echo "No local SSD disks found."
    fi
  done
}

# Installs logrotate configuration files
function setup-logrotate() {
  mkdir -p /etc/logrotate.d/
  cat >/etc/logrotate.d/docker-containers <<EOF
/var/lib/docker/containers/*/*-json.log {
    rotate 5
    copytruncate
    missingok
    notifempty
    compress
    maxsize 10M
    daily
    dateext
    dateformat -%Y%m%d-%s
    create 0644 root root
}
EOF

  # Configure log rotation for all logs in /var/log, which is where k8s services
  # are configured to write their log files. Whenever logrotate is ran, this
  # config will:
  # * rotate the log file if its size is > 100Mb OR if one day has elapsed
  # * save rotated logs into a gzipped timestamped backup
  # * log file timestamp (controlled by 'dateformat') includes seconds too. this
  #   ensures that logrotate can generate unique logfiles during each rotation
  #   (otherwise it skips rotation if 'maxsize' is reached multiple times in a
  #   day).
  # * keep only 5 old (rotated) logs, and will discard older logs.
  cat > /etc/logrotate.d/allvarlogs <<EOF
/var/log/*.log {
    rotate 5
    copytruncate
    missingok
    notifempty
    compress
    maxsize 100M
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

# After the first boot and on upgrade, these files exist on the master-pd
# and should never be touched again (except perhaps an additional service
# account, see NB below.)
function create-master-auth {
  echo "Creating master auth files"
  local -r auth_dir="/etc/srv/kubernetes"
  if [[ ! -e "${auth_dir}/ca.crt" && ! -z "${CA_CERT:-}" && ! -z "${MASTER_CERT:-}" && ! -z "${MASTER_KEY:-}" ]]; then
    echo "${CA_CERT}" | base64 --decode > "${auth_dir}/ca.crt"
    echo "${MASTER_CERT}" | base64 --decode > "${auth_dir}/server.cert"
    echo "${MASTER_KEY}" | base64 --decode > "${auth_dir}/server.key"
  fi
  local -r basic_auth_csv="${auth_dir}/basic_auth.csv"
  if [[ ! -e "${basic_auth_csv}" ]]; then
    echo "${KUBE_PASSWORD},${KUBE_USER},admin" > "${basic_auth_csv}"
  fi
  local -r known_tokens_csv="${auth_dir}/known_tokens.csv"
  if [[ ! -e "${known_tokens_csv}" ]]; then
    echo "${KUBE_BEARER_TOKEN},admin,admin" > "${known_tokens_csv}"
    echo "${KUBELET_TOKEN},kubelet,kubelet" >> "${known_tokens_csv}"
    echo "${KUBE_PROXY_TOKEN},kube_proxy,kube_proxy" >> "${known_tokens_csv}"
  fi
  local use_cloud_config="false"
  cat <<EOF >/etc/gce.conf
[global]
EOF
  if [[ -n "${PROJECT_ID:-}" && -n "${TOKEN_URL:-}" && -n "${TOKEN_BODY:-}" && -n "${NODE_NETWORK:-}" ]]; then
    use_cloud_config="true"
    cat <<EOF >>/etc/gce.conf
token-url = ${TOKEN_URL}
token-body = ${TOKEN_BODY}
project-id = ${PROJECT_ID}
network-name = ${NODE_NETWORK}
EOF
  fi
  if [[ -n "${NODE_INSTANCE_PREFIX:-}" ]]; then
    use_cloud_config="true"
    if [[ -n "${NODE_TAGS:-}" ]]; then
      local -r node_tags="${NODE_TAGS}"
    else
      local -r node_tags="${NODE_INSTANCE_PREFIX}"
    fi
    cat <<EOF >>/etc/gce.conf
node-tags = ${node_tags}
node-instance-prefix = ${NODE_INSTANCE_PREFIX}
EOF
  fi
  if [[ -n "${MULTIZONE:-}" ]]; then
    use_cloud_config="true"
    cat <<EOF >>/etc/gce.conf
multizone = ${MULTIZONE}
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

function create-kubelet-kubeconfig {
  echo "Creating kubelet kubeconfig file"
  if [[ -z "${KUBELET_CA_CERT:-}" ]]; then
    KUBELET_CA_CERT="${CA_CERT}"
  fi
  cat <<EOF >/var/lib/kubelet/kubeconfig
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
}

# Uses KUBELET_CA_CERT (falling back to CA_CERT), KUBELET_CERT, and KUBELET_KEY
# to generate a kubeconfig file for the kubelet to securely connect to the apiserver.
function create-master-kubelet-auth {
  # Only configure the kubelet on the master if the required variables are
  # set in the environment.
  if [[ -n "${KUBELET_APISERVER:-}" && -n "${KUBELET_CERT:-}" && -n "${KUBELET_KEY:-}" ]]; then
    create-kubelet-kubeconfig
  fi
}

function create-kubeproxy-kubeconfig {
  echo "Creating kube-proxy kubeconfig file"
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
    certificate-authority-data: ${CA_CERT}
contexts:
- context:
    cluster: local
    user: kube-proxy
  name: service-account-context
current-context: service-account-context
EOF
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

  echo "DOCKER_OPTS=\"${docker_opts} ${EXTRA_DOCKER_OPTS:-}\"" > /etc/default/docker
  # If using a network plugin, we need to explicitly restart docker daemon, because
  # kubelet will not do it.
  if [[ "${use_net_plugin}" == "true" ]]; then
    echo "Docker command line is updated. Restart docker to pick it up"
    systemctl restart docker
  fi
}

# A helper function for loading a docker image. It keeps trying up to 5 times.
#
# $1: Full path of the docker image
function try-load-docker-image {
  local -r img=$1
  echo "Try to load docker image file ${img}"
  # Temporarily turn off errexit, because we don't want to exit on first failure.
  set +e
  local -r max_attempts=5
  local -i attempt_num=1
  until timeout 30 docker load -i "${img}"; do
    if [[ "${attempt_num}" == "${max_attempts}" ]]; then
      echo "Fail to load docker image file ${img} after ${max_attempts} retries. Exist!!"
      exit 1
    else
      attempt_num=$((attempt_num+1))
      sleep 5
    fi
  done
  # Re-enable errexit.
  set -e
}

# Loads kube-system docker images. It is better to do it before starting kubelet,
# as kubelet will restart docker daemon, which may interfere with loading images.
function load-docker-images {
  echo "Start loading kube-system docker images"
  local -r img_dir="${KUBE_HOME}/kube-docker-files"
  if [[ "${KUBERNETES_MASTER:-}" == "true" ]]; then
    try-load-docker-image "${img_dir}/kube-apiserver.tar"
    try-load-docker-image "${img_dir}/kube-controller-manager.tar"
    try-load-docker-image "${img_dir}/kube-scheduler.tar"
  else
    try-load-docker-image "${img_dir}/kube-proxy.tar"
  fi
}

# This function assembles the kubelet systemd service file and starts it
# using systemctl.
function start-kubelet {
  echo "Start kubelet"
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
  flags+=" --babysit-daemons=true"
  flags+=" --cgroup-root=/"
  flags+=" --cloud-provider=gce"
  flags+=" --cluster-dns=${DNS_SERVER_IP}"
  flags+=" --cluster-domain=${DNS_DOMAIN}"
  flags+=" --config=/etc/kubernetes/manifests"
  flags+=" --kubelet-cgroups=/kubelet"
  flags+=" --system-cgroups=/system"

  if [[ -n "${KUBELET_PORT:-}" ]]; then
    flags+=" --port=${KUBELET_PORT}"
  fi
  local reconcile_cidr="true"
  if [[ "${KUBERNETES_MASTER:-}" == "true" ]]; then
    flags+=" --enable-debugging-handlers=false"
    flags+=" --hairpin-mode=none"
    if [[ ! -z "${KUBELET_APISERVER:-}" && ! -z "${KUBELET_CERT:-}" && ! -z "${KUBELET_KEY:-}" ]]; then
      flags+=" --api-servers=https://${KUBELET_APISERVER}"
      flags+=" --register-schedulable=false"
      # need at least a /29 pod cidr for now due to #32844
      # TODO: determine if we still allow non-hostnetwork pods to run on master, clean up master pod setup
      # WARNING: potential ip range collision with 10.123.45.0/29
      flags+=" --pod-cidr=10.123.45.0/29"
      reconcile_cidr="false"
    else
      flags+=" --pod-cidr=${MASTER_IP_RANGE}"
    fi
  else # For nodes
    flags+=" --enable-debugging-handlers=true"
    flags+=" --api-servers=https://${KUBERNETES_MASTER_NAME}"
    if [[ "${HAIRPIN_MODE:-}" == "promiscuous-bridge" ]] || \
       [[ "${HAIRPIN_MODE:-}" == "hairpin-veth" ]] || \
       [[ "${HAIRPIN_MODE:-}" == "none" ]]; then
      flags+=" --hairpin-mode=${HAIRPIN_MODE}"
    fi
  fi
  # Network plugin
  if [[ -n "${NETWORK_PROVIDER:-}" ]]; then
    if [[ "${NETWORK_PROVIDER:-}" == "cni" ]]; then
      flags+=" --cni-bin-dir=/home/kubernetes/bin"
    else
      flags+=" --network-plugin-dir=/home/kubernetes/bin"
    fi
    flags+=" --network-plugin=${NETWORK_PROVIDER}"
  fi
  flags+=" --reconcile-cidr=${reconcile_cidr}"
  if [[ -n "${NON_MASQUERADE_CIDR:-}" ]]; then
    flag+=" --non-masquerade-cidr=${NON_MASQUERADE_CIDR}"
  fi
  if [[ "${ENABLE_MANIFEST_URL:-}" == "true" ]]; then
    flags+=" --manifest-url=${MANIFEST_URL}"
    flags+=" --manifest-url-header=${MANIFEST_URL_HEADER}"
  fi
  if [[ -n "${ENABLE_CUSTOM_METRICS:-}" ]]; then
    flags+=" --enable-custom-metrics=${ENABLE_CUSTOM_METRICS}"
  fi
  if [[ -n "${NODE_LABELS:-}" ]]; then
    flags+=" --node-labels=${NODE_LABELS}"
  fi
  if [[ -n "${EVICTION_HARD:-}" ]]; then
    flags+=" --eviction-hard=${EVICTION_HARD}"
  fi
  if [[ "${ALLOCATE_NODE_CIDRS:-}" == "true" ]]; then
     flags+=" --configure-cbr0=${ALLOCATE_NODE_CIDRS}"
  fi
  if [[ -n "${FEATURE_GATES:-}" ]]; then
     flags+=" --feature-gates=${FEATURE_GATES}"
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

  # Flush iptables nat table
  iptables -t nat -F || true

  systemctl start kubelet.service
}

# Create the log file and set its properties.
#
# $1 is the file to create.
function prepare-log-file {
  touch $1
  chmod 644 $1
  chown root:root $1
}

# Starts kube-proxy pod.
function start-kube-proxy {
  echo "Start kube-proxy pod"
  prepare-log-file /var/log/kube-proxy.log
  local -r src_file="${KUBE_HOME}/kube-manifests/kubernetes/kube-proxy.manifest"
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
  if [[ -n "${KUBEPROXY_TEST_ARGS:-}" ]]; then
    params+=" ${KUBE_PROXY_TEST_ARGS}"
  fi
  sed -i -e "s@{{kubeconfig}}@${kubeconfig}@g" ${src_file}
  sed -i -e "s@{{pillar\['kube_docker_registry'\]}}@${kube_docker_registry}@g" ${src_file}
  sed -i -e "s@{{pillar\['kube-proxy_docker_tag'\]}}@${kube_proxy_docker_tag}@g" ${src_file}
  sed -i -e "s@{{params}}@${params}@g" ${src_file}
  sed -i -e "s@{{ cpurequest }}@100m@g" ${src_file}
  sed -i -e "s@{{api_servers_with_port}}@${api_servers}@g" ${src_file}
  if [[ -n "${CLUSTER_IP_RANGE:-}" ]]; then
    sed -i -e "s@{{cluster_cidr}}@--cluster-cidr=${CLUSTER_IP_RANGE}@g" ${src_file}
  fi
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
  local host_name=$(hostname)
  local etcd_cluster=""
  local cluster_state="new"
  for host in $(echo "${INITIAL_ETCD_CLUSTER:-${host_name}}" | tr "," "\n"); do
    etcd_host="etcd-${host}=http://${host}:$3"
    if [[ -n "${etcd_cluster}" ]]; then
      etcd_cluster+=","
      cluster_state="existing"
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
  sed -i -e "s@{{ *etcd_cluster *}}@$etcd_cluster@g" "${temp_file}"
  sed -i -e "s@{{ *storage_backend *}}@${STORAGE_BACKEND:-}@g" "${temp_file}"
  sed -i -e "s@{{ *cluster_state *}}@$cluster_state@g" "${temp_file}"
  if [[ -n "${TEST_ETCD_VERSION:-}" ]]; then
    sed -i -e "s@{{ *pillar\.get('etcd_docker_tag', '\(.*\)') *}}@${TEST_ETCD_VERSION}@g" "${temp_file}"
  else
    sed -i -e "s@{{ *pillar\.get('etcd_docker_tag', '\(.*\)') *}}@\1@g" "${temp_file}"
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
function compute-master-manifest-variables {
  CLOUD_CONFIG_OPT=""
  CLOUD_CONFIG_VOLUME=""
  CLOUD_CONFIG_MOUNT=""
  if [[ -f /etc/gce.conf ]]; then
    CLOUD_CONFIG_OPT="--cloud-config=/etc/gce.conf"
    CLOUD_CONFIG_VOLUME="{\"name\": \"cloudconfigmount\",\"hostPath\": {\"path\": \"/etc/gce.conf\"}},"
    CLOUD_CONFIG_MOUNT="{\"name\": \"cloudconfigmount\",\"mountPath\": \"/etc/gce.conf\", \"readOnly\": true},"
  fi
  DOCKER_REGISTRY="gcr.io/google_containers"
  if [[ -n "${KUBE_DOCKER_REGISTRY:-}" ]]; then
    DOCKER_REGISTRY="${KUBE_DOCKER_REGISTRY}"
  fi
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

  # Calculate variables and assemble the command line.
  local params="${API_SERVER_TEST_LOG_LEVEL:-"--v=2"} ${APISERVER_TEST_ARGS:-} ${CLOUD_CONFIG_OPT}"
  params+=" --address=127.0.0.1"
  params+=" --allow-privileged=true"
  params+=" --authorization-policy-file=/etc/srv/kubernetes/abac-authz-policy.jsonl"
  params+=" --basic-auth-file=/etc/srv/kubernetes/basic_auth.csv"
  params+=" --cloud-provider=gce"
  params+=" --client-ca-file=/etc/srv/kubernetes/ca.crt"
  params+=" --etcd-servers=http://127.0.0.1:2379"
  params+=" --etcd-servers-overrides=/events#http://127.0.0.1:4002"
  params+=" --secure-port=443"
  params+=" --tls-cert-file=/etc/srv/kubernetes/server.cert"
  params+=" --tls-private-key-file=/etc/srv/kubernetes/server.key"
  params+=" --token-auth-file=/etc/srv/kubernetes/known_tokens.csv"
  if [[ -n "${STORAGE_BACKEND:-}" ]]; then
    params+=" --storage-backend=${STORAGE_BACKEND}"
  fi
  if [[ -n "${ENABLE_GARBAGE_COLLECTOR:-}" ]]; then
    params+=" --enable-garbage-collector=${ENABLE_GARBAGE_COLLECTOR}"
  fi
  if [[ -n "${NUM_NODES:-}" ]]; then
    # Set amount of memory available for apiserver based on number of nodes.
    # TODO: Once we start setting proper requests and limits for apiserver
    # we should reuse the same logic here instead of current heuristic.
    params+=" --target-ram-mb=$((${NUM_NODES} * 60))"
  fi
  if [[ -n "${SERVICE_CLUSTER_IP_RANGE:-}" ]]; then
    params+=" --service-cluster-ip-range=${SERVICE_CLUSTER_IP_RANGE}"
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
      admission_controller_config_volume="{\"name\": \"admissioncontrollerconfigmount\",\"hostPath\": {\"path\": \"/etc/admission_controller.config\"}},"
      # Mount the file to configure the ImagePolicyWebhook's webhook.
      image_policy_webhook_config_mount="{\"name\": \"imagepolicywebhookconfigmount\",\"mountPath\": \"/etc/gcp_image_review.config\", \"readOnly\": false},"
      image_policy_webhook_config_volume="{\"name\": \"imagepolicywebhookconfigmount\",\"hostPath\": {\"path\": \"/etc/gcp_image_review.config\"}},"
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
    local -r vm_external_ip=$(curl --retry 5 --retry-delay 3 --fail --silent -H 'Metadata-Flavor: Google' "http://metadata/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip")
    params+=" --advertise-address=${vm_external_ip}"
    params+=" --ssh-user=${PROXY_SSH_USER}"
    params+=" --ssh-keyfile=/etc/srv/sshproxy/.sshkeyfile"
  fi

  local webhook_authn_config_mount=""
  local webhook_authn_config_volume=""
  if [[ -n "${GCP_AUTHN_URL:-}" ]]; then
    params+=" --authentication-token-webhook-config-file=/etc/gcp_authn.config"
    webhook_authn_config_mount="{\"name\": \"webhookauthnconfigmount\",\"mountPath\": \"/etc/gcp_authn.config\", \"readOnly\": false},"
    webhook_authn_config_volume="{\"name\": \"webhookauthnconfigmount\",\"hostPath\": {\"path\": \"/etc/gcp_authn.config\"}},"
  fi

  params+=" --authorization-mode=ABAC"
  local webhook_config_mount=""
  local webhook_config_volume=""
  if [[ -n "${GCP_AUTHZ_URL:-}" ]]; then
    params+=",Webhook --authorization-webhook-config-file=/etc/gcp_authz.config"
    webhook_config_mount="{\"name\": \"webhookconfigmount\",\"mountPath\": \"/etc/gcp_authz.config\", \"readOnly\": false},"
    webhook_config_volume="{\"name\": \"webhookconfigmount\",\"hostPath\": {\"path\": \"/etc/gcp_authz.config\"}},"
  fi
  local -r src_dir="${KUBE_HOME}/kube-manifests/kubernetes/gci-trusty"

  if [[ -n "${KUBE_USER:-}" ]]; then
    local -r abac_policy_json="${src_dir}/abac-authz-policy.jsonl"
    remove-salt-config-comments "${abac_policy_json}"
    sed -i -e "s/{{kube_user}}/${KUBE_USER}/g" "${abac_policy_json}"
    cp "${abac_policy_json}" /etc/srv/kubernetes/
  fi

  src_file="${src_dir}/kube-apiserver.manifest"
  remove-salt-config-comments "${src_file}"
  # Evaluate variables.
  local -r kube_apiserver_docker_tag=$(cat /home/kubernetes/kube-docker-files/kube-apiserver.docker_tag)
  sed -i -e "s@{{params}}@${params}@g" "${src_file}"
  sed -i -e "s@{{srv_kube_path}}@/etc/srv/kubernetes@g" "${src_file}"
  sed -i -e "s@{{srv_sshproxy_path}}@/etc/srv/sshproxy@g" "${src_file}"
  sed -i -e "s@{{cloud_config_mount}}@${CLOUD_CONFIG_MOUNT}@g" "${src_file}"
  sed -i -e "s@{{cloud_config_volume}}@${CLOUD_CONFIG_VOLUME}@g" "${src_file}"
  sed -i -e "s@{{pillar\['kube_docker_registry'\]}}@${DOCKER_REGISTRY}@g" "${src_file}"
  sed -i -e "s@{{pillar\['kube-apiserver_docker_tag'\]}}@${kube_apiserver_docker_tag}@g" "${src_file}"
  sed -i -e "s@{{pillar\['allow_privileged'\]}}@true@g" "${src_file}"
  sed -i -e "s@{{secure_port}}@443@g" "${src_file}"
  sed -i -e "s@{{secure_port}}@8080@g" "${src_file}"
  sed -i -e "s@{{additional_cloud_config_mount}}@@g" "${src_file}"
  sed -i -e "s@{{additional_cloud_config_volume}}@@g" "${src_file}"
  sed -i -e "s@{{webhook_authn_config_mount}}@${webhook_authn_config_mount}@g" "${src_file}"
  sed -i -e "s@{{webhook_authn_config_volume}}@${webhook_authn_config_volume}@g" "${src_file}"
  sed -i -e "s@{{webhook_config_mount}}@${webhook_config_mount}@g" "${src_file}"
  sed -i -e "s@{{webhook_config_volume}}@${webhook_config_volume}@g" "${src_file}"
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
  prepare-log-file /var/log/kube-controller-manager.log
  # Calculate variables and assemble the command line.
  local params="${CONTROLLER_MANAGER_TEST_LOG_LEVEL:-"--v=2"} ${CONTROLLER_MANAGER_TEST_ARGS:-} ${CLOUD_CONFIG_OPT}"
  params+=" --cloud-provider=gce"
  params+=" --master=127.0.0.1:8080"
  params+=" --root-ca-file=/etc/srv/kubernetes/ca.crt"
  params+=" --service-account-private-key-file=/etc/srv/kubernetes/server.key"
  if [[ -n "${ENABLE_GARBAGE_COLLECTOR:-}" ]]; then
    params+=" --enable-garbage-collector=${ENABLE_GARBAGE_COLLECTOR}"
  fi
  if [[ -n "${INSTANCE_PREFIX:-}" ]]; then
    params+=" --cluster-name=${INSTANCE_PREFIX}"
  fi
  if [[ -n "${CLUSTER_IP_RANGE:-}" ]]; then
    params+=" --cluster-cidr=${CLUSTER_IP_RANGE}"
  fi
  if [[ -n "${SERVICE_CLUSTER_IP_RANGE:-}" ]]; then
    params+=" --service-cluster-ip-range=${SERVICE_CLUSTER_IP_RANGE}"
  fi
  if [[ "${NETWORK_PROVIDER:-}" == "kubenet" ]]; then
    params+=" --allocate-node-cidrs=true"
  elif [[ -n "${ALLOCATE_NODE_CIDRS:-}" ]]; then
    params+=" --allocate-node-cidrs=${ALLOCATE_NODE_CIDRS}"
  fi
  if [[ -n "${TERMINATED_POD_GC_THRESHOLD:-}" ]]; then
    params+=" --terminated-pod-gc-threshold=${TERMINATED_POD_GC_THRESHOLD}"
  fi
  if [[ -n "${FEATURE_GATES:-}" ]]; then
    params+=" --feature-gates=${FEATURE_GATES}"
  fi
  local -r kube_rc_docker_tag=$(cat /home/kubernetes/kube-docker-files/kube-controller-manager.docker_tag)

  local -r src_file="${KUBE_HOME}/kube-manifests/kubernetes/gci-trusty/kube-controller-manager.manifest"
  remove-salt-config-comments "${src_file}"
  # Evaluate variables.
  sed -i -e "s@{{srv_kube_path}}@/etc/srv/kubernetes@g" "${src_file}"
  sed -i -e "s@{{pillar\['kube_docker_registry'\]}}@${DOCKER_REGISTRY}@g" "${src_file}"
  sed -i -e "s@{{pillar\['kube-controller-manager_docker_tag'\]}}@${kube_rc_docker_tag}@g" "${src_file}"
  sed -i -e "s@{{params}}@${params}@g" "${src_file}"
  sed -i -e "s@{{cloud_config_mount}}@${CLOUD_CONFIG_MOUNT}@g" "${src_file}"
  sed -i -e "s@{{cloud_config_volume}}@${CLOUD_CONFIG_VOLUME}@g" "${src_file}"
  sed -i -e "s@{{additional_cloud_config_mount}}@@g" "${src_file}"
  sed -i -e "s@{{additional_cloud_config_volume}}@@g" "${src_file}"
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
  prepare-log-file /var/log/kube-scheduler.log

  # Calculate variables and set them in the manifest.
  params="${SCHEDULER_TEST_LOG_LEVEL:-"--v=2"} ${SCHEDULER_TEST_ARGS:-}"
  if [[ -n "${FEATURE_GATES:-}" ]]; then
    params+=" --feature-gates=${FEATURE_GATES}"
  fi
  if [[ -n "${SCHEDULING_ALGORITHM_PROVIDER:-}"  ]]; then
    params+=" --algorithm-provider=${SCHEDULING_ALGORITHM_PROVIDER}"
  fi
  local -r kube_scheduler_docker_tag=$(cat "${KUBE_HOME}/kube-docker-files/kube-scheduler.docker_tag")

  # Remove salt comments and replace variables with values.
  local -r src_file="${KUBE_HOME}/kube-manifests/kubernetes/gci-trusty/kube-scheduler.manifest"
  remove-salt-config-comments "${src_file}"

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

    local params="${AUTOSCALER_MIG_CONFIG} ${CLOUD_CONFIG_OPT}"
    sed -i -e "s@{{params}}@${params}@g" "${src_file}"
    sed -i -e "s@{{cloud_config_mount}}@${CLOUD_CONFIG_MOUNT}@g" "${src_file}"
    sed -i -e "s@{{cloud_config_volume}}@${CLOUD_CONFIG_VOLUME}@g" "${src_file}"
    sed -i -e "s@{%.*%}@@g" "${src_file}"

    cp "${src_file}" /etc/kubernetes/manifests
  fi
}

# A helper function for copying addon manifests and set dir/files
# permissions.
#
# $1: addon category under /etc/kubernetes
# $2: manifest source dir
function setup-addon-manifests {
  local -r src_dir="${KUBE_HOME}/kube-manifests/kubernetes/gci-trusty/$2"
  local -r dst_dir="/etc/kubernetes/$1/$2"
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

# Prepares the manifests of k8s addons, and starts the addon manager.
function start-kube-addons {
  echo "Prepare kube-addons manifests and start kube addon manager"
  local -r src_dir="${KUBE_HOME}/kube-manifests/kubernetes/gci-trusty"
  local -r dst_dir="/etc/kubernetes/addons"
  # Set up manifests of other addons.
  if [[ "${ENABLE_CLUSTER_MONITORING:-}" == "influxdb" ]] || \
     [[ "${ENABLE_CLUSTER_MONITORING:-}" == "google" ]] || \
     [[ "${ENABLE_CLUSTER_MONITORING:-}" == "standalone" ]] || \
     [[ "${ENABLE_CLUSTER_MONITORING:-}" == "googleinfluxdb" ]]; then
    local -r file_dir="cluster-monitoring/${ENABLE_CLUSTER_MONITORING}"
    setup-addon-manifests "addons" "${file_dir}"
    # Replace the salt configurations with variable values.
    base_metrics_memory="140Mi"
    metrics_memory="${base_metrics_memory}"
    base_eventer_memory="190Mi"
    base_metrics_cpu="80m"
    metrics_cpu="${base_metrics_cpu}"
    eventer_memory="${base_eventer_memory}"
    nanny_memory="90Mi"
    local -r metrics_memory_per_node="4"
    local -r metrics_cpu_per_node="0.5"
    local -r eventer_memory_per_node="500"
    local -r nanny_memory_per_node="200"
    if [[ -n "${NUM_NODES:-}" && "${NUM_NODES}" -ge 1 ]]; then
      num_kube_nodes="$((${NUM_NODES}+1))"
      metrics_memory="$((${num_kube_nodes} * ${metrics_memory_per_node} + 200))Mi"
      eventer_memory="$((${num_kube_nodes} * ${eventer_memory_per_node} + 200 * 1024))Ki"
      nanny_memory="$((${num_kube_nodes} * ${nanny_memory_per_node} + 90 * 1024))Ki"
      metrics_cpu=$(echo - | awk "{print ${num_kube_nodes} * ${metrics_cpu_per_node} + 80}")m
    fi
    controller_yaml="${dst_dir}/${file_dir}"
    if [[ "${ENABLE_CLUSTER_MONITORING:-}" == "googleinfluxdb" ]]; then
      controller_yaml="${controller_yaml}/heapster-controller-combined.yaml"
    else
      controller_yaml="${controller_yaml}/heapster-controller.yaml"
    fi
    remove-salt-config-comments "${controller_yaml}"
    sed -i -e "s@{{ *base_metrics_memory *}}@${base_metrics_memory}@g" "${controller_yaml}"
    sed -i -e "s@{{ *metrics_memory *}}@${metrics_memory}@g" "${controller_yaml}"
    sed -i -e "s@{{ *base_metrics_cpu *}}@${base_metrics_cpu}@g" "${controller_yaml}"
    sed -i -e "s@{{ *metrics_cpu *}}@${metrics_cpu}@g" "${controller_yaml}"
    sed -i -e "s@{{ *base_eventer_memory *}}@${base_eventer_memory}@g" "${controller_yaml}"
    sed -i -e "s@{{ *eventer_memory *}}@${eventer_memory}@g" "${controller_yaml}"
    sed -i -e "s@{{ *metrics_memory_per_node *}}@${metrics_memory_per_node}@g" "${controller_yaml}"
    sed -i -e "s@{{ *eventer_memory_per_node *}}@${eventer_memory_per_node}@g" "${controller_yaml}"
    sed -i -e "s@{{ *nanny_memory *}}@${nanny_memory}@g" "${controller_yaml}"
    sed -i -e "s@{{ *metrics_cpu_per_node *}}@${metrics_cpu_per_node}@g" "${controller_yaml}"
  fi
  if [[ "${ENABLE_CLUSTER_DNS:-}" == "true" ]]; then
    setup-addon-manifests "addons" "dns"
    local -r dns_rc_file="${dst_dir}/dns/skydns-rc.yaml"
    local -r dns_svc_file="${dst_dir}/dns/skydns-svc.yaml"
    mv "${dst_dir}/dns/skydns-rc.yaml.in" "${dns_rc_file}"
    mv "${dst_dir}/dns/skydns-svc.yaml.in" "${dns_svc_file}"
    # Replace the salt configurations with variable values.
    sed -i -e "s@{{ *pillar\['dns_replicas'\] *}}@${DNS_REPLICAS}@g" "${dns_rc_file}"
    sed -i -e "s@{{ *pillar\['dns_domain'\] *}}@${DNS_DOMAIN}@g" "${dns_rc_file}"
    sed -i -e "s@{{ *pillar\['dns_server'\] *}}@${DNS_SERVER_IP}@g" "${dns_svc_file}"

    if [[ "${FEDERATION:-}" == "true" ]]; then
      local federations_domain_map="${FEDERATIONS_DOMAIN_MAP:-}"
      if [[ -z "${federations_domain_map}" && -n "${FEDERATION_NAME:-}" && -n "${DNS_ZONE_NAME:-}" ]]; then
        federations_domain_map="${FEDERATION_NAME}=${DNS_ZONE_NAME}"
      fi
      if [[ -n "${federations_domain_map}" ]]; then
        sed -i -e "s@{{ *pillar\['federations_domain_map'\] *}}@- --federations=${federations_domain_map}@g" "${dns_rc_file}"
      else
        sed -i -e "/{{ *pillar\['federations_domain_map'\] *}}/d" "${dns_rc_file}"
      fi
    else
      sed -i -e "/{{ *pillar\['federations_domain_map'\] *}}/d" "${dns_rc_file}"
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
  if [[ "${ENABLE_CLUSTER_UI:-}" == "true" ]]; then
    setup-addon-manifests "addons" "dashboard"
  fi
  if [[ "${ENABLE_NODE_PROBLEM_DETECTOR:-}" == "true" ]]; then
    setup-addon-manifests "addons" "node-problem-detector"
  fi
  if echo "${ADMISSION_CONTROL:-}" | grep -q "LimitRanger"; then
    setup-addon-manifests "admission-controls" "limit-range"
  fi
  if [[ "${NETWORK_POLICY_PROVIDER:-}" == "calico" ]]; then
    setup-addon-manifests "addons" "calico-policy-controller"
  fi

  # Place addon manager pod manifest.
  cp "${src_dir}/kube-addon-manager.yaml" /etc/kubernetes/manifests
}

# Starts a fluentd static pod for logging.
function start-fluentd {
  echo "Start fluentd pod"
  if [[ "${ENABLE_NODE_LOGGING:-}" == "true" ]]; then
    if [[ "${LOGGING_DESTINATION:-}" == "gcp" ]]; then
      cp "${KUBE_HOME}/kube-manifests/kubernetes/gci-trusty/gci/fluentd-gcp.yaml" /etc/kubernetes/manifests/
    elif [[ "${LOGGING_DESTINATION:-}" == "elasticsearch" && "${KUBERNETES_MASTER:-}" != "true" ]]; then
      # Running fluentd-es on the master is pointless, as it can't communicate
      # with elasticsearch from there in the default configuration.
      cp "${KUBE_HOME}/kube-manifests/kubernetes/fluentd-es.yaml" /etc/kubernetes/manifests/
    fi
  fi
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
    cp "${KUBE_HOME}/kube-manifests/kubernetes/gci-trusty/glbc.manifest" \
       /etc/kubernetes/manifests/
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
    mount --bind /var/lib/kubelet /var/lib/kubelet/
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


########### Main Function ###########
echo "Start to configure instance for kubernetes"

KUBE_HOME="/home/kubernetes"
if [[ ! -e "${KUBE_HOME}/kube-env" ]]; then
  echo "The ${KUBE_HOME}/kube-env file does not exist!! Terminate cluster initialization."
  exit 1
fi

source "${KUBE_HOME}/kube-env"

if [[ -n "${KUBE_USER:-}" ]]; then
  if ! [[ "${KUBE_USER}" =~ ^[-._@a-zA-Z0-9]+$ ]]; then
    echo "Bad KUBE_USER format."
    exit 1
  fi
fi

config-ip-firewall
create-dirs
setup-kubelet-dir
ensure-local-ssds
setup-logrotate
if [[ "${KUBERNETES_MASTER:-}" == "true" ]]; then
  mount-master-pd
  create-master-auth
  create-master-kubelet-auth
else
  create-kubelet-kubeconfig
  create-kubeproxy-kubeconfig
fi

assemble-docker-flags
load-docker-images
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
  start-kube-proxy
  # Kube-registry-proxy.
  if [[ "${ENABLE_CLUSTER_REGISTRY:-}" == "true" ]]; then
    start-kube-registry-proxy
  fi
  if [[ "${PREPULL_E2E_IMAGES:-}" == "true" ]]; then
    start-image-puller
  fi
fi
start-fluentd
reset-motd
echo "Done for the configuration for kubernetes"
