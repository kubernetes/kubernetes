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
# files, or the manifest files should not be templated salt

set -o errexit
set -o nounset
set -o pipefail

function create-dirs {
  echo "Creating required directories"
  mkdir -p /var/lib/kubelet
  mkdir -p /etc/kubernetes/manifests
  if [[ "${KUBERNETES_MASTER:-}" == "false" ]]; then
    mkdir -p /var/lib/kube-proxy
  fi
}

# Create directories referenced in the kube-controller-manager manifest for
# bindmounts. This is used under the rkt runtime to work around
# https://github.com/kubernetes/kubernetes/issues/26816
function create-kube-controller-manager-dirs {
  mkdir -p /etc/srv/kubernetes /var/ssl /etc/{ssl,openssl,pki}
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

# replace_prefixed_line ensures:
# 1. the specified file exists
# 2. existing lines with the specified ${prefix} are removed
# 3. a new line with the specified ${prefix}${suffix} is appended
function replace_prefixed_line {
  local -r file="${1:-}"
  local -r prefix="${2:-}"
  local -r suffix="${3:-}"

  touch "${file}"
  awk "substr(\$0,0,length(\"${prefix}\")) != \"${prefix}\" { print }" "${file}" > "${file}.filtered"  && mv "${file}.filtered" "${file}"
  echo "${prefix}${suffix}" >> "${file}"
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
  if [[ -n "${KUBE_PASSWORD:-}" && -n "${KUBE_USER:-}" ]]; then
    replace_prefixed_line "${basic_auth_csv}" "${KUBE_PASSWORD},${KUBE_USER}," "admin,system:masters"
  fi
  local -r known_tokens_csv="${auth_dir}/known_tokens.csv"
  if [[ -n "${KUBE_BEARER_TOKEN:-}" ]]; then
    replace_prefixed_line "${known_tokens_csv}" "${KUBE_BEARER_TOKEN},"             "admin,admin,system:masters"
  fi
  if [[ -n "${KUBE_CONTROLLER_MANAGER_TOKEN:-}" ]]; then
    replace_prefixed_line "${known_tokens_csv}" "${KUBE_CONTROLLER_MANAGER_TOKEN}," "system:kube-controller-manager,uid:system:kube-controller-manager"
  fi
  if [[ -n "${KUBE_SCHEDULER_TOKEN:-}" ]]; then
    replace_prefixed_line "${known_tokens_csv}" "${KUBE_SCHEDULER_TOKEN},"          "system:kube-scheduler,uid:system:kube-scheduler"
  fi
  if [[ -n "${KUBELET_TOKEN:-}" ]]; then
    replace_prefixed_line "${known_tokens_csv}" "${KUBELET_TOKEN},"                 "kubelet,uid:kubelet,system:nodes"
  fi
  if [[ -n "${KUBE_PROXY_TOKEN:-}" ]]; then
    replace_prefixed_line "${known_tokens_csv}" "${KUBE_PROXY_TOKEN},"              "system:kube-proxy,uid:kube_proxy"
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
  if [[ -n "${GCE_ALPHA_FEATURES:-}" ]]; then
    use_cloud_config="true"
    cat <<EOF >>/etc/gce.conf
alpha-features = ${GCE_ALPHA_FEATURES}
EOF
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

# Arg 1: the address of the API server
function create-kubelet-kubeconfig() {
  local apiserver_address="${1}"
  if [[ -z "${apiserver_address}" ]]; then
    echo "Must provide API server address to create Kubelet kubeconfig file!"
    exit 1
  fi
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
    server: ${apiserver_address}
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
# Set REGISTER_MASTER_KUBELET to true if kubelet on the master node
# should register to the apiserver.
function create-master-kubelet-auth {
  # Only configure the kubelet on the master if the required variables are
  # set in the environment.
  if [[ -n "${KUBELET_APISERVER:-}" && -n "${KUBELET_CERT:-}" && -n "${KUBELET_KEY:-}" ]]; then
    REGISTER_MASTER_KUBELET="true"
    create-kubelet-kubeconfig "https://${KUBELET_APISERVER}"
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
    certificate-authority-data: ${CA_CERT}
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

function create-master-etcd-auth {
  if [[ -n "${ETCD_CA_CERT:-}" && -n "${ETCD_PEER_KEY:-}" && -n "${ETCD_PEER_CERT:-}" ]]; then
    local -r auth_dir="/etc/srv/kubernetes"
    echo "${ETCD_CA_CERT}" | base64 --decode | gunzip > "${auth_dir}/etcd-ca.crt"
    echo "${ETCD_PEER_KEY}" | base64 --decode > "${auth_dir}/etcd-peer.key"
    echo "${ETCD_PEER_CERT}" | base64 --decode | gunzip > "${auth_dir}/etcd-peer.crt"
  fi
}

function configure-docker-daemon {
  echo "Configuring the Docker daemon"
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

  mkdir -p /etc/systemd/system/docker.service.d/
  local kubernetes_conf_dropin="/etc/systemd/system/docker.service.d/00_kubelet.conf"
  cat > "${kubernetes_conf_dropin}" <<EOF
[Service]
Environment="DOCKER_OPTS=${docker_opts} ${EXTRA_DOCKER_OPTS:-}"
EOF
  # Always restart to get the cbr0 change
  echo "Docker daemon options updated. Restarting docker..."
  systemctl daemon-reload
  systemctl restart docker
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

  if [[ "${CONTAINER_RUNTIME:-}" == "rkt" ]]; then
    for attempt_num in $(seq 1 "${max_attempts}"); do
      local aci_tmpdir="$(mktemp -t -d docker2aci.XXXXX)"
      (cd "${aci_tmpdir}"; timeout 40 "${DOCKER2ACI_BIN}" "$1")
      local aci_success=$?
      timeout 40 "${RKT_BIN}" fetch --insecure-options=image "${aci_tmpdir}"/*.aci
      local fetch_success=$?
      rm -f "${aci_tmpdir}"/*.aci
      rmdir "${aci_tmpdir}"
      if [[ ${fetch_success} && ${aci_success} ]]; then
        echo "rkt: Loaded ${img}"
        break
      fi
      if [[ "${attempt}" == "${max_attempts}" ]]; then
        echo "rkt: Failed to load image file ${img} after ${max_attempts} retries."
        exit 1
      fi
      sleep 5
    done
  else
    until timeout 30 docker load -i "${img}"; do
      if [[ "${attempt_num}" == "${max_attempts}" ]]; then
        echo "Fail to load docker image file ${img} after ${max_attempts} retries."
        exit 1
      else
        attempt_num=$((attempt_num+1))
        sleep 5
      fi
    done
  fi
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
  echo "Using kubelet binary at ${kubelet_bin}"
  local flags="${KUBELET_TEST_LOG_LEVEL:-"--v=2"} ${KUBELET_TEST_ARGS:-}"
  flags+=" --allow-privileged=true"
  flags+=" --cgroup-root=/"
  flags+=" --cloud-provider=gce"
  flags+=" --cluster-dns=${DNS_SERVER_IP}"
  flags+=" --cluster-domain=${DNS_DOMAIN}"
  flags+=" --pod-manifest-path=/etc/kubernetes/manifests"
  flags+=" --experimental-check-node-capabilities-before-mount=true"

  if [[ -n "${KUBELET_PORT:-}" ]]; then
    flags+=" --port=${KUBELET_PORT}"
  fi
  if [[ "${KUBERNETES_MASTER:-}" == "true" ]]; then
    flags+=" --enable-debugging-handlers=false"
    flags+=" --hairpin-mode=none"
    if [[ "${REGISTER_MASTER_KUBELET:-false}" == "true" ]]; then
      flags+=" --kubeconfig=/var/lib/kubelet/kubeconfig"
      flags+=" --register-schedulable=false"
    else
      # Standalone mode (not widely used?)
      flags+=" --pod-cidr=${MASTER_IP_RANGE}"
    fi
  else # For nodes
    flags+=" --enable-debugging-handlers=true"
    flags+=" --kubeconfig=/var/lib/kubelet/kubeconfig"
    if [[ "${HAIRPIN_MODE:-}" == "promiscuous-bridge" ]] || \
       [[ "${HAIRPIN_MODE:-}" == "hairpin-veth" ]] || \
       [[ "${HAIRPIN_MODE:-}" == "none" ]]; then
      flags+=" --hairpin-mode=${HAIRPIN_MODE}"
    fi
  fi
  # Network plugin
  if [[ -n "${NETWORK_PROVIDER:-}" ]]; then
    flags+=" --cni-bin-dir=/opt/kubernetes/bin"
    flags+=" --network-plugin=${NETWORK_PROVIDER}"
  fi
  if [[ -n "${NON_MASQUERADE_CIDR:-}" ]]; then
    flags+=" --non-masquerade-cidr=${NON_MASQUERADE_CIDR}"
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
  if [[ -n "${CONTAINER_RUNTIME:-}" ]]; then
    flags+=" --container-runtime=${CONTAINER_RUNTIME}"
    flags+=" --rkt-path=${KUBE_HOME}/bin/rkt"
    flags+=" --rkt-stage1-image=${RKT_STAGE1_IMAGE}"
  fi

  local -r kubelet_env_file="/etc/kubelet-env"
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
  local -r kube_proxy_docker_tag=$(cat /opt/kubernetes/kube-docker-files/kube-proxy.docker_tag)
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
  if [[ "${CONTAINER_RUNTIME:-}" == "rkt" ]]; then
    # Work arounds for https://github.com/coreos/rkt/issues/3245 and https://github.com/coreos/rkt/issues/3264
    # This is an incredibly hacky workaround. It's fragile too. If the kube-proxy command changes too much, this breaks
    # TODO, this could be done much better in many other places, such as an
    # init script within the container, or even within kube-proxy's code.
    local extra_workaround_cmd="ln -sf /proc/self/mounts /etc/mtab; \
      mount -o remount,rw /proc; \
      mount -o remount,rw /proc/sys; \
      mount -o remount,rw /sys; "
    sed -i -e "s@-\\s\\+kube-proxy@- ${extra_workaround_cmd} kube-proxy@g" "${src_file}"
  fi
}

# Starts kube-proxy static pod.
function start-kube-proxy {
  echo "Start kube-proxy static pod"
  prepare-log-file /var/log/kube-proxy.log
  local -r src_file="${KUBE_HOME}/kube-manifests/kubernetes/kube-proxy.manifest"
  prepare-kube-proxy-manifest-variables "$src_file"

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

  if [[ -n "${ETCD_CA_KEY:-}" && -n "${ETCD_CA_CERT:-}" && -n "${ETCD_PEER_KEY:-}" && -n "${ETCD_PEER_CERT:-}" ]]; then
    etcd_creds=" --peer-trusted-ca-file /etc/srv/kubernetes/etcd-ca.crt --peer-cert-file /etc/srv/kubernetes/etcd-peer.crt --peer-key-file /etc/srv/kubernetes/etcd-peer.key -peer-client-cert-auth "
    etcd_protocol="https"
  fi

  for host in $(echo "${INITIAL_ETCD_CLUSTER:-${host_name}}" | tr "," "\n"); do
    etcd_host="etcd-${host}=${etcd_protocol}://${host}:$3"
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
  sed -i -e "s@{{ *srv_kube_path *}}@/etc/srv/kubernetes@g" "${temp_file}"
  sed -i -e "s@{{ *etcd_cluster *}}@$etcd_cluster@g" "${temp_file}"
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
    sed -i -e "s@{{ *quota_bytes *}}@--quota-backend-bytes=4294967296@g" "${temp_file}"
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
  params+=" --client-ca-file=/etc/srv/kubernetes/ca.crt"
  params+=" --etcd-servers=http://127.0.0.1:2379"
  params+=" --etcd-servers-overrides=/events#http://127.0.0.1:4002"
  params+=" --secure-port=443"
  params+=" --tls-cert-file=/etc/srv/kubernetes/server.cert"
  params+=" --tls-private-key-file=/etc/srv/kubernetes/server.key"
  params+=" --token-auth-file=/etc/srv/kubernetes/known_tokens.csv"
  params+=" --enable-aggregator-routing=true"
  if [[ -n "${KUBE_PASSWORD:-}" && -n "${KUBE_USER:-}" ]]; then
    params+=" --basic-auth-file=/etc/srv/kubernetes/basic_auth.csv"
  fi
  if [[ -n "${STORAGE_BACKEND:-}" ]]; then
    params+=" --storage-backend=${STORAGE_BACKEND}"
  fi
  if [[ -n "${STORAGE_MEDIA_TYPE:-}" ]]; then
    params+=" --storage-media-type=${STORAGE_MEDIA_TYPE}"
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
    local -r vm_external_ip=$(curl --retry 5 --retry-delay 3 --fail --silent -H 'Metadata-Flavor: Google' "http://metadata/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip")
    if [[ -n "${PROXY_SSH_USER:-}" ]]; then
      params+=" --advertise-address=${vm_external_ip}"      
      params+=" --ssh-user=${PROXY_SSH_USER}"
      params+=" --ssh-keyfile=/etc/srv/sshproxy/.sshkeyfile"
    else
      params+=" --kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname",
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

  local authorization_mode="RBAC"
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

  src_file="${src_dir}/kube-apiserver.manifest"
  remove-salt-config-comments "${src_file}"
  # Evaluate variables.
  local -r kube_apiserver_docker_tag=$(cat /opt/kubernetes/kube-docker-files/kube-apiserver.docker_tag)
  sed -i -e "s@{{params}}@${params}@g" "${src_file}"
  sed -i -e "s@{{container_env}}@${container_env}@g" "${src_file}"
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
  create-kubecontrollermanager-kubeconfig
  prepare-log-file /var/log/kube-controller-manager.log
  # Calculate variables and assemble the command line.
  local params="${CONTROLLER_MANAGER_TEST_LOG_LEVEL:-"--v=2"} ${CONTROLLER_MANAGER_TEST_ARGS:-} ${CLOUD_CONFIG_OPT}"
  params+=" --use-service-account-credentials"
  params+=" --cloud-provider=gce"
  params+=" --kubeconfig=/etc/srv/kubernetes/kube-controller-manager/kubeconfig"
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
    params+=" --cidr-allocator-type=CloudAllocator"
    params+=" --configure-cloud-routes=false"
  fi
  if [[ -n "${FEATURE_GATES:-}" ]]; then
    params+=" --feature-gates=${FEATURE_GATES}"
  fi
  local -r kube_rc_docker_tag=$(cat /opt/kubernetes/kube-docker-files/kube-controller-manager.docker_tag)
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
  sed -i -e "s@{{container_env}}@${container_env}@g" "${src_file}"
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

# Prepares the manifests of k8s addons, and starts the addon manager.
# Vars assumed:
#   CLUSTER_NAME
function start-kube-addons {
  echo "Prepare kube-addons manifests and start kube addon manager"
  local -r src_dir="${KUBE_HOME}/kube-manifests/kubernetes/gci-trusty"
  local -r dst_dir="/etc/kubernetes/addons"

  # prep addition kube-up specific rbac objects
  setup-addon-manifests "addons" "rbac"

  # Set up manifests of other addons.
  if [[ "${KUBE_PROXY_DAEMONSET:-}" == "true" ]]; then
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
  fi
  if [[ "${ENABLE_METRICS_SERVER:-}" == "true" ]]; then
    setup-addon-manifests "addons" "metrics-server"
  fi
  if [[ "${ENABLE_CLUSTER_DNS:-}" == "true" ]]; then
    setup-addon-manifests "addons" "dns"
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
  fi
  if [[ "${ENABLE_CLUSTER_UI:-}" == "true" ]]; then
    setup-addon-manifests "addons" "dashboard"
  fi
  if [[ "${ENABLE_NODE_PROBLEM_DETECTOR:-}" == "daemonset" ]]; then
    setup-addon-manifests "addons" "node-problem-detector"
  fi
  if echo "${ADMISSION_CONTROL:-}" | grep -q "LimitRanger"; then
    setup-addon-manifests "admission-controls" "limit-range"
  fi
  if [[ "${NETWORK_POLICY_PROVIDER:-}" == "calico" ]]; then
    setup-addon-manifests "addons" "calico-policy-controller"

    # Configure Calico CNI directory.
    local -r ds_file="${dst_dir}/calico-policy-controller/calico-node-daemonset.yaml"
    sed -i -e "s@__CALICO_CNI_DIR__@/opt/cni/bin@g" "${ds_file}"
  fi
  if [[ "${ENABLE_DEFAULT_STORAGE_CLASS:-}" == "true" ]]; then
    setup-addon-manifests "addons" "storage-class/gce"
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

# Install and setup rkt
# TODO(euank): There should be a toggle to use the distro-provided rkt binary
# Sets the following variables:
#   RKT_BIN: the path to the rkt binary
function setup-rkt {
    local rkt_bin="${KUBE_HOME}/bin/rkt"
    if [[ -x "${rkt_bin}" ]]; then
      # idempotency, skip downloading this time
      # TODO(euank): this might get in the way of updates, but 'file busy'
      # because of rkt-api would too
      RKT_BIN="${rkt_bin}"
      return
    fi
    mkdir -p /etc/rkt "${KUBE_HOME}/download/"
    local rkt_tar="${KUBE_HOME}/download/rkt.tar.gz"
    local rkt_tmpdir=$(mktemp -d "${KUBE_HOME}/rkt_download.XXXXX")
    curl --retry 5 --retry-delay 3 --fail --silent --show-error \
      --location --create-dirs --output "${rkt_tar}" \
      https://github.com/coreos/rkt/releases/download/v${RKT_VERSION}/rkt-v${RKT_VERSION}.tar.gz
    tar --strip-components=1 -xf "${rkt_tar}" -C "${rkt_tmpdir}" --overwrite
    mv "${rkt_tmpdir}/rkt" "${rkt_bin}"
    if [[ ! -x "${rkt_bin}" ]]; then
      echo "Could not download requested rkt binary"
      exit 1
    fi
    RKT_BIN="${rkt_bin}"
    # Cache rkt stage1 images for speed
    "${RKT_BIN}" fetch --insecure-options=image "${rkt_tmpdir}"/*.aci
    rm -rf "${rkt_tmpdir}"

    cat > /etc/systemd/system/rkt-api.service <<EOF
[Unit]
Description=rkt api service
Documentation=http://github.com/coreos/rkt
After=network.target

[Service]
ExecStart=${RKT_BIN} api-service --listen=127.0.0.1:15441
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF
    systemctl enable rkt-api.service
    systemctl start rkt-api.service
}

# Install docker2aci, needed to load server images if using rkt runtime
# This should be removed once rkt can fetch on-disk docker tarballs directly
# Sets the following variables:
#   DOCKER2ACI_BIN: the path to the docker2aci binary
function install-docker2aci {
  local tar_path="${KUBE_HOME}/download/docker2aci.tar.gz"
  local tmp_path="${KUBE_HOME}/docker2aci"
  mkdir -p "${KUBE_HOME}/download/" "${tmp_path}"
  curl --retry 5 --retry-delay 3 --fail --silent --show-error \
    --location --create-dirs --output "${tar_path}" \
    https://github.com/appc/docker2aci/releases/download/v0.14.0/docker2aci-v0.14.0.tar.gz
  tar --strip-components=1 -xf "${tar_path}" -C "${tmp_path}" --overwrite
  DOCKER2ACI_BIN="${KUBE_HOME}/bin/docker2aci"
  mv "${tmp_path}/docker2aci" "${DOCKER2ACI_BIN}"
}

########### Main Function ###########
echo "Start to configure instance for kubernetes"

# Note: this name doesn't make as much sense here as in gci where it's actually
# /home/kubernetes, but for ease of diff-ing, retain the same variable name
KUBE_HOME="/opt/kubernetes"
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

# generate the controller manager and scheduler tokens here since they are only used on the master.
KUBE_CONTROLLER_MANAGER_TOKEN=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
KUBE_SCHEDULER_TOKEN=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)

# KUBERNETES_CONTAINER_RUNTIME is set by the `kube-env` file, but it's a bit of a mouthful
if [[ "${CONTAINER_RUNTIME:-}" == "" ]]; then
  CONTAINER_RUNTIME="${KUBERNETES_CONTAINER_RUNTIME:-docker}"
fi

create-dirs
ensure-local-ssds
if [[ "${KUBERNETES_MASTER:-}" == "true" ]]; then
  mount-master-pd
  create-master-auth
  create-master-kubelet-auth
  create-master-etcd-auth
else
  create-kubelet-kubeconfig "https://${KUBERNETES_MASTER_NAME}"
  if [[ "${KUBE_PROXY_DAEMONSET:-}" != "true" ]]; then
    create-kubeproxy-user-kubeconfig
  fi
fi

if [[ "${CONTAINER_RUNTIME:-}" == "rkt" ]]; then
  systemctl stop docker
  systemctl disable docker
  setup-rkt
  install-docker2aci
  create-kube-controller-manager-dirs
else
  configure-docker-daemon
fi

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
fi
echo "Done for the configuration for kubernetes"
