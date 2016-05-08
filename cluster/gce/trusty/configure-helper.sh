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

# This script contains functions for configuring instances to run kubernetes
# master and nodes. It is uploaded in kube-manifests tarball, and used by
# upstart jobs in cluster/gce/trusty/<master, node>.yaml


config_hostname() {
  # Set the hostname to the short version.
  host_name=$(curl --silent "http://metadata.google.internal/computeMetadata/v1/instance/hostname" -H "Metadata-Flavor: Google")
  short_hostname=$(echo "${host_name}" | cut -d. -f1)
  hostname "${short_hostname}"
}

config_ip_firewall() {
  # We have seen that GCE image may have strict host firewall rules which drop
  # most inbound/forwarded packets. In such a case, add rules to accept all
  # TCP/UDP packets.
  if iptables -L INPUT | grep "Chain INPUT (policy DROP)" > /dev/null; then
    echo "Add rules to accpet all inbound TCP/UDP packets"
    iptables -A INPUT -w -p TCP -j ACCEPT
    iptables -A INPUT -w -p UDP -j ACCEPT
  fi
  if iptables -L FORWARD | grep "Chain FORWARD (policy DROP)" > /dev/null; then
    echo "Add rules to accpet all forwarded TCP/UDP packets"
    iptables -A FORWARD -w -p TCP -j ACCEPT
    iptables -A FORWARD -w -p UDP -j ACCEPT
  fi
}

create_dirs() {
  # Create required directories.
  mkdir -p /var/lib/kubelet
  mkdir -p /etc/kubernetes/manifests
  if [ "${KUBERNETES_MASTER:-}" = "false" ]; then
    mkdir -p /var/lib/kube-proxy
  fi
}

create_kubelet_kubeconfig() {
  # Create the kubelet kubeconfig file.
  if [ -z "${KUBELET_CA_CERT:-}" ]; then
    KUBELET_CA_CERT="${CA_CERT}"
  fi
  cat > /var/lib/kubelet/kubeconfig << EOF
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

create_kubeproxy_kubeconfig() {
  # Create the kube-proxy config file.
  cat > /var/lib/kube-proxy/kubeconfig << EOF
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

# Installs the critical packages that are required by spinning up a cluster.
install_critical_packages() {
  apt-get update
  # Install docker and brctl if they are not in the image.
  if ! which docker > /dev/null; then
    echo "Do not find docker. Install it."
    curl -fsSL https://get.docker.com/ | sh
  fi
  if ! which brctl > /dev/null; then
    echo "Do not find brctl. Install it."
    apt-get install --yes bridge-utils
  fi
}

# Install the packages that are useful but not required by spinning up a cluster.
install_additional_packages() {
  if ! which socat > /dev/null; then
    echo "Do not find socat. Install it."
    apt-get install --yes socat
  fi
  if ! which nsenter > /dev/null; then
    echo "Do not find nsenter. Install it."
    mkdir -p /tmp/nsenter-install
    cd /tmp/nsenter-install
    curl https://www.kernel.org/pub/linux/utils/util-linux/v2.24/util-linux-2.24.tar.gz | tar -zxf-
    apt-get --yes install make
    apt-get --yes install gcc
    cd util-linux-2.24
    ./configure --without-ncurses
    make nsenter
    cp nsenter /usr/local/bin
    rm -rf /tmp/nsenter-install
  fi
}

# Assembles kubelet command line flags.
# It should be called by master and nodes before running kubelet process. The caller
# needs to source the config file /etc/kube-env. This function sets the following
# variable that will be used as the kubelet command line flags
#   KUBELET_CMD_FLAGS
assemble_kubelet_flags() {
  KUBELET_CMD_FLAGS="--v=2"
  if [ -n "${KUBELET_TEST_LOG_LEVEL:-}" ]; then
    KUBELET_CMD_FLAGS="${KUBELET_TEST_LOG_LEVEL}"
  fi
  if [ -n "${KUBELET_PORT:-}" ]; then
    KUBELET_CMD_FLAGS="${KUBELET_CMD_FLAGS} --port=${KUBELET_PORT}"
  fi
  if [ -n "${KUBELET_TEST_ARGS:-}" ]; then
    KUBELET_CMD_FLAGS="${KUBELET_CMD_FLAGS} ${KUBELET_TEST_ARGS}"
  fi
  if [ "${KUBERNETES_MASTER:-}" = "true" ]; then
    KUBELET_CMD_FLAGS="${KUBELET_CMD_FLAGS} --enable-debugging-handlers=false --hairpin-mode=none"
    if [ ! -z "${KUBELET_APISERVER:-}" ] && \
       [ ! -z "${KUBELET_CERT:-}" ] && \
       [ ! -z "${KUBELET_KEY:-}" ]; then
      KUBELET_CMD_FLAGS="${KUBELET_CMD_FLAGS} --api-servers=https://${KUBELET_APISERVER} --register-schedulable=false --reconcile-cidr=false --pod-cidr=10.123.45.0/30"
    else
      KUBELET_CMD_FLAGS="${KUBELET_CMD_FLAGS} --pod-cidr=${MASTER_IP_RANGE}"
    fi
  else # For nodes
    KUBELET_CMD_FLAGS="${KUBELET_CMD_FLAGS} --enable-debugging-handlers=true --api-servers=https://${KUBERNETES_MASTER_NAME}"
    if [ "${HAIRPIN_MODE:-}" = "promiscuous-bridge" ] || \
       [ "${HAIRPIN_MODE:-}" = "hairpin-veth" ] || \
       [ "${HAIRPIN_MODE:-}" = "none" ]; then
      KUBELET_CMD_FLAGS="${KUBELET_CMD_FLAGS} --hairpin-mode=${HAIRPIN_MODE}"
    fi
  fi
  if [ "${ENABLE_MANIFEST_URL:-}" = "true" ]; then
    KUBELET_CMD_FLAGS="${KUBELET_CMD_FLAGS} --manifest-url=${MANIFEST_URL} --manifest-url-header=${MANIFEST_URL_HEADER}"
  fi
  if [ -n "${ENABLE_CUSTOM_METRICS:-}" ]; then
    KUBELET_CMD_FLAGS="${KUBELET_CMD_FLAGS} --enable-custom-metrics=${ENABLE_CUSTOM_METRICS}"
  fi
  if [ -n "${NODE_LABELS:-}" ]; then
    KUBELET_CMD_FLAGS="${KUBELET_CMD_FLAGS} --node-labels=${NODE_LABELS}"
  fi
  if [ "${ALLOCATE_NODE_CIDRS:-}" = "true" ]; then
     KUBELET_CMD_FLAGS="${KUBELET_CMD_FLAGS} --configure-cbr0=${ALLOCATE_NODE_CIDRS}"
  fi
  # Add the unconditional flags
  KUBELET_CMD_FLAGS="${KUBELET_CMD_FLAGS} --cloud-provider=gce --allow-privileged=true --cgroup-root=/ --system-cgroups=/system --kubelet-cgroups=/kubelet --babysit-daemons=true --config=/etc/kubernetes/manifests --cluster-dns=${DNS_SERVER_IP} --cluster-domain=${DNS_DOMAIN}"
  echo "KUBELET_OPTS=\"${KUBELET_CMD_FLAGS}\"" > /etc/default/kubelet
}

restart_docker_daemon() {
  DOCKER_OPTS="-p /var/run/docker.pid --bridge=cbr0 --iptables=false --ip-masq=false"
  if [ "${TEST_CLUSTER:-}" = "true" ]; then
    DOCKER_OPTS="${DOCKER_OPTS} --debug"
  fi
  echo "DOCKER_OPTS=\"${DOCKER_OPTS} ${EXTRA_DOCKER_OPTS:-}\"" > /etc/default/docker
  # Make sure the network interface cbr0 is created before restarting docker daemon
  while ! [ -L /sys/class/net/cbr0 ]; do
    echo "Sleep 1 second to wait for cbr0"
    sleep 1
  done
  # Remove docker0
  ifconfig docker0 down
  brctl delbr docker0
  # Ensure docker daemon is really functional before exiting. Operations afterwards may
  # assume it is running.
  while ! docker version > /dev/null; do
    echo "Sleep 1 second to wait for docker daemon"
    sleep 1
  done
}

# Create the log file and set its properties.
#
# $1 is the file to create
prepare_log_file() {
  touch $1
  chmod 644 $1
  chown root:root $1
}

# It monitors the health of several master and node components.
health_monitoring() {
  sleep_seconds=10
  max_seconds=10
  # We simply kill the process when there is a failure. Another upstart job will automatically
  # restart the process.
  while [ 1 ]; do
    if ! timeout 10 docker version > /dev/null; then
      echo "Docker daemon failed!"
      pkill docker
      # Wait for a while, as we don't want to kill it again before it is really up.
      sleep 30
    fi
    if ! curl --insecure -m "${max_seconds}" -f -s https://127.0.0.1:${KUBELET_PORT:-10250}/healthz > /dev/null; then
      echo "Kubelet is unhealthy!"
      pkill kubelet
      # Wait for a while, as we don't want to kill it again before it is really up.
      sleep 60
    fi
    sleep "${sleep_seconds}"
  done
}


########## The functions below are for master only ##########

# Mounts a persistent disk (formatting if needed) to store the persistent data
# on the master -- etcd's data, a few settings, and security certs/keys/tokens.
# safe_format_and_mount only formats an unformatted disk, and mkdir -p will
# leave a directory be if it already exists.
mount_master_pd() {
  readonly pd_path="/dev/disk/by-id/google-master-pd"
  readonly mount_point="/mnt/disks/master-pd"

  # TODO(zmerlynn): GKE is still lagging in master-pd creation
  if [ ! -e "${pd_path}" ]; then
    return
  fi
  # Format and mount the disk, create directories on it for all of the master's
  # persistent data, and link them to where they're used.
  mkdir -p "${mount_point}"
  /usr/share/google/safe_format_and_mount -m "mkfs.ext4 -F" "${pd_path}" "${mount_point}" >/var/log/master-pd-mount.log || \
    { echo "!!! master-pd mount failed, review /var/log/master-pd-mount.log !!!"; return 1; }
  # Contains all the data stored in etcd
  mkdir -m 700 -p "${mount_point}/var/etcd"
  # Contains the dynamically generated apiserver auth certs and keys
  mkdir -p "${mount_point}/etc/srv/kubernetes"
  # Directory for kube-apiserver to store SSH key (if necessary)
  mkdir -p "${mount_point}/etc/srv/sshproxy"
  ln -s -f "${mount_point}/var/etcd" /var/etcd
  mkdir -p /etc/srv
  ln -s -f "${mount_point}/etc/srv/kubernetes" /etc/srv/kubernetes
  ln -s -f "${mount_point}/etc/srv/sshproxy" /etc/srv/sshproxy

  if ! id etcd &>/dev/null; then
    useradd -s /sbin/nologin -d /var/etcd etcd
  fi
  chown -R etcd "${mount_point}/var/etcd"
  chgrp -R etcd "${mount_point}/var/etcd"
}

# After the first boot and on upgrade, these files exists on the master-pd
# and should never be touched again (except perhaps an additional service
# account, see NB below.)
create_master_auth() {
  readonly auth_dir="/etc/srv/kubernetes"
  if [ ! -e "${auth_dir}/ca.crt" ]; then
    if  [ ! -z "${CA_CERT:-}" ] && [ ! -z "${MASTER_CERT:-}" ] && [ ! -z "${MASTER_KEY:-}" ]; then
      echo "${CA_CERT}" | base64 -d > "${auth_dir}/ca.crt"
      echo "${MASTER_CERT}" | base64 -d > "${auth_dir}/server.cert"
      echo "${MASTER_KEY}" | base64 -d > "${auth_dir}/server.key"
      # Kubecfg cert/key are optional and included for backwards compatibility.
      # TODO(roberthbailey): Remove these two lines once GKE no longer requires
      # fetching clients certs from the master VM.
      echo "${KUBECFG_CERT:-}" | base64 -d > "${auth_dir}/kubecfg.crt"
      echo "${KUBECFG_KEY:-}" | base64 -d > "${auth_dir}/kubecfg.key"
    fi
  fi
  readonly basic_auth_csv="${auth_dir}/basic_auth.csv"
  if [ ! -e "${basic_auth_csv}" ]; then
    echo "${KUBE_PASSWORD},${KUBE_USER},admin" > "${basic_auth_csv}"
  fi
  readonly known_tokens_csv="${auth_dir}/known_tokens.csv"
  if [ ! -e "${known_tokens_csv}" ]; then
    echo "${KUBE_BEARER_TOKEN},admin,admin" > "${known_tokens_csv}"
    echo "${KUBELET_TOKEN},kubelet,kubelet" >> "${known_tokens_csv}"
    echo "${KUBE_PROXY_TOKEN},kube_proxy,kube_proxy" >> "${known_tokens_csv}"
  fi

  if [ -n "${PROJECT_ID:-}" ] && [ -n "${TOKEN_URL:-}" ] && [ -n "${TOKEN_BODY:-}" ] && [ -n "${NODE_NETWORK:-}" ]; then
    cat <<EOF >/etc/gce.conf
[global]
token-url = ${TOKEN_URL}
token-body = ${TOKEN_BODY}
project-id = ${PROJECT_ID}
network-name = ${NODE_NETWORK}
EOF
  fi
  if [ -n "${MULTIZONE:-}" ]; then
    cat <<EOF >>/etc/gce.conf
multizone = ${MULTIZONE}
EOF
  fi
}

# Uses KUBELET_CA_CERT (falling back to CA_CERT), KUBELET_CERT, and KUBELET_KEY
# to generate a kubeconfig file for the kubelet to securely connect to the apiserver.
create_master_kubelet_auth() {
  # Only configure the kubelet on the master if the required variables are
  # set in the environment.
  if [ -n "${KUBELET_APISERVER:-}" ] && [ -n "${KUBELET_CERT:-}" ] && [ -n "${KUBELET_KEY:-}" ]; then
    create_kubelet_kubeconfig
  fi
}

# Replaces the variables in the etcd manifest file with the real values, and then
# copy the file to the manifest dir
# $1: value for variable 'suffix'
# $2: value for variable 'port'
# $3: value for variable 'server_port'
# $4: value for variable 'cpulimit'
# $5: pod name, which should be either etcd or etcd-events
prepare_etcd_manifest() {
  etcd_temp_file="/tmp/$5"
  cp /home/kubernetes/kube-manifests/kubernetes/gci-trusty/etcd.manifest "${etcd_temp_file}"
  sed -i -e "s@{{ *suffix *}}@$1@g" "${etcd_temp_file}"
  sed -i -e "s@{{ *port *}}@$2@g" "${etcd_temp_file}"
  sed -i -e "s@{{ *server_port *}}@$3@g" "${etcd_temp_file}"
  sed -i -e "s@{{ *cpulimit *}}@\"$4\"@g" "${etcd_temp_file}"
  # Replace the volume host path
  sed -i -e "s@/mnt/master-pd/var/etcd@/mnt/disks/master-pd/var/etcd@g" "${etcd_temp_file}"
  mv "${etcd_temp_file}" /etc/kubernetes/manifests
}

# Starts etcd server pod (and etcd-events pod if needed).
# More specifically, it prepares dirs and files, sets the variable value
# in the manifests, and copies them to /etc/kubernetes/manifests.
start_etcd_servers() {
  if [ -d /etc/etcd ]; then
    rm -rf /etc/etcd
  fi
  if [ -e /etc/default/etcd ]; then
    rm -f /etc/default/etcd
  fi
  if [ -e /etc/systemd/system/etcd.service ]; then
    rm -f /etc/systemd/system/etcd.service
  fi
  if [ -e /etc/init.d/etcd ]; then
    rm -f /etc/init.d/etcd
  fi
  prepare_log_file /var/log/etcd.log
  prepare_etcd_manifest "" "4001" "2380" "200m" "etcd.manifest"

  prepare_log_file /var/log/etcd-events.log
  prepare_etcd_manifest "-events" "4002" "2381" "100m" "etcd-events.manifest"
}

# Calculates the following variables based on env variables, which will be used
# by the manifests of several kube-master components.
#   CLOUD_CONFIG_VOLUME
#   CLOUD_CONFIG_MOUNT
#   DOCKER_REGISTRY
compute_master_manifest_variables() {
  CLOUD_CONFIG_VOLUME=""
  CLOUD_CONFIG_MOUNT=""
  if [ -n "${PROJECT_ID:-}" ] && [ -n "${TOKEN_URL:-}" ] && [ -n "${TOKEN_BODY:-}" ] && [ -n "${NODE_NETWORK:-}" ]; then
    CLOUD_CONFIG_VOLUME="{\"name\": \"cloudconfigmount\",\"hostPath\": {\"path\": \"/etc/gce.conf\"}},"
    CLOUD_CONFIG_MOUNT="{\"name\": \"cloudconfigmount\",\"mountPath\": \"/etc/gce.conf\", \"readOnly\": true},"
  fi
  DOCKER_REGISTRY="gcr.io/google_containers"
  if [ -n "${KUBE_DOCKER_REGISTRY:-}" ]; then
    DOCKER_REGISTRY="${KUBE_DOCKER_REGISTRY}"
  fi
}

# A helper function for removing salt configuration and comments from a file.
# This is mainly for preparing a manifest file.
# $1: Full path of the file to manipulate
remove_salt_config_comments() {
  # Remove salt configuration
  sed -i "/^[ |\t]*{[#|%]/d" $1
  # Remove comments
  sed -i "/^[ |\t]*#/d" $1
}

# Starts k8s apiserver.
# It prepares the log file, loads the docker image, calculates variables, sets them
# in the manifest file, and then copies the manifest file to /etc/kubernetes/manifests.
#
# Assumed vars (which are calculated in function compute_master_manifest_variables)
#   CLOUD_CONFIG_VOLUME
#   CLOUD_CONFIG_MOUNT
#   DOCKER_REGISTRY
start_kube_apiserver() {
  prepare_log_file /var/log/kube-apiserver.log
  # Load the docker image from file.
  echo "Try to load docker image file kube-apiserver.tar"
  timeout 30 docker load -i /home/kubernetes/kube-docker-files/kube-apiserver.tar

  # Calculate variables and assemble the command line.
  params="--cloud-provider=gce --address=127.0.0.1 --etcd-servers=http://127.0.0.1:4001 --tls-cert-file=/etc/srv/kubernetes/server.cert --tls-private-key-file=/etc/srv/kubernetes/server.key --secure-port=443 --client-ca-file=/etc/srv/kubernetes/ca.crt --token-auth-file=/etc/srv/kubernetes/known_tokens.csv --basic-auth-file=/etc/srv/kubernetes/basic_auth.csv --allow-privileged=true --authorization-mode=ABAC --authorization-policy-file=/etc/srv/kubernetes/abac-authz-policy.jsonl --etcd-servers-overrides=/events#http://127.0.0.1:4002"
  if [ -n "${SERVICE_CLUSTER_IP_RANGE:-}" ]; then
    params="${params} --service-cluster-ip-range=${SERVICE_CLUSTER_IP_RANGE}"
  fi
  if [ -n "${ADMISSION_CONTROL:-}" ]; then
    params="${params} --admission-control=${ADMISSION_CONTROL}"
  fi
  if [ -n "${KUBE_APISERVER_REQUEST_TIMEOUT:-}" ]; then
    params="${params} --min-request-timeout=${KUBE_APISERVER_REQUEST_TIMEOUT}"
  fi
  if [ -n "${RUNTIME_CONFIG:-}" ]; then
    params="${params} --runtime-config=${RUNTIME_CONFIG}"
  fi
  if [ -n "${APISERVER_TEST_ARGS:-}" ]; then
    params="${params} ${APISERVER_TEST_ARGS}"
  fi
  log_level="--v=2"
  if [ -n "${API_SERVER_TEST_LOG_LEVEL:-}" ]; then
    log_level="${API_SERVER_TEST_LOG_LEVEL}"
  fi
  params="${params} ${log_level}"

  if [ -n "${PROJECT_ID:-}" ] && [ -n "${TOKEN_URL:-}" ] && [ -n "${TOKEN_BODY:-}" ] && [ -n "${NODE_NETWORK:-}" ]; then
    readonly vm_external_ip=$(curl --fail --silent -H 'Metadata-Flavor: Google' "http://metadata/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip")
    params="${params} --cloud-config=/etc/gce.conf --advertise-address=${vm_external_ip} --ssh-user=${PROXY_SSH_USER} --ssh-keyfile=/etc/srv/sshproxy/.sshkeyfile"
  fi
  readonly kube_apiserver_docker_tag=$(cat /home/kubernetes/kube-docker-files/kube-apiserver.docker_tag)

  src_dir="/home/kubernetes/kube-manifests/kubernetes/gci-trusty"
  cp "${src_dir}/abac-authz-policy.jsonl" /etc/srv/kubernetes/
  src_file="${src_dir}/kube-apiserver.manifest"
  remove_salt_config_comments "${src_file}"
  # Evaluate variables
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
  cp "${src_file}" /etc/kubernetes/manifests
}

# Starts k8s controller manager.
# It prepares the log file, loads the docker image, calculates variables, sets them
# in the manifest file, and then copies the manifest file to /etc/kubernetes/manifests.
#
# Assumed vars (which are calculated in function compute_master_manifest_variables)
#   CLOUD_CONFIG_VOLUME
#   CLOUD_CONFIG_MOUNT
#   DOCKER_REGISTRY
start_kube_controller_manager() {
  prepare_log_file /var/log/kube-controller-manager.log
  # Load the docker image from file.
  echo "Try to load docker image file kube-controller-manager.tar"
  timeout 30 docker load -i /home/kubernetes/kube-docker-files/kube-controller-manager.tar

  # Calculate variables and assemble the command line.
  params="--master=127.0.0.1:8080 --cloud-provider=gce --root-ca-file=/etc/srv/kubernetes/ca.crt --service-account-private-key-file=/etc/srv/kubernetes/server.key"
  if [ -n "${PROJECT_ID:-}" ] && [ -n "${TOKEN_URL:-}" ] && [ -n "${TOKEN_BODY:-}" ] && [ -n "${NODE_NETWORK:-}" ]; then
    params="${params} --cloud-config=/etc/gce.conf"
  fi
  if [ -n "${INSTANCE_PREFIX:-}" ]; then
    params="${params} --cluster-name=${INSTANCE_PREFIX}"
  fi
  if [ -n "${CLUSTER_IP_RANGE:-}" ]; then
    params="${params} --cluster-cidr=${CLUSTER_IP_RANGE}"
  fi
  if [ "${ALLOCATE_NODE_CIDRS:-}" = "true" ]; then
    params="${params} --allocate-node-cidrs=${ALLOCATE_NODE_CIDRS}"
  fi
  if [ -n "${TERMINATED_POD_GC_THRESHOLD:-}" ]; then
    params="${params} --terminated-pod-gc-threshold=${TERMINATED_POD_GC_THRESHOLD}"
  fi
  log_level="--v=2"
  if [ -n "${CONTROLLER_MANAGER_TEST_LOG_LEVEL:-}" ]; then
    log_level="${CONTROLLER_MANAGER_TEST_LOG_LEVEL}"
  fi
  params="${params} ${log_level}"
  if [ -n "${CONTROLLER_MANAGER_TEST_ARGS:-}" ]; then
    params="${params} ${CONTROLLER_MANAGER_TEST_ARGS}"
  fi
  readonly kube_rc_docker_tag=$(cat /home/kubernetes/kube-docker-files/kube-controller-manager.docker_tag)

  src_file="/home/kubernetes/kube-manifests/kubernetes/gci-trusty/kube-controller-manager.manifest"
  remove_salt_config_comments "${src_file}"
  # Evaluate variables
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

# Starts k8s scheduler.
# It prepares the log file, loads the docker image, calculates variables, sets them
# in the manifest file, and then copies the manifest file to /etc/kubernetes/manifests.
#
# Assumed vars (which are calculated in compute_master_manifest_variables())
#   DOCKER_REGISTRY
start_kube_scheduler() {
  prepare_log_file /var/log/kube-scheduler.log
  kube_home="home/kubernetes"
  # Load the docker image from file.
  echo "Try to load docker image file kube-scheduler.tar"
  timeout 30 docker load -i "${kube_home}/kube-docker-files/kube-scheduler.tar"

  # Calculate variables and set them in the manifest.
  params=""
  log_level="--v=2"
  if [ -n "${SCHEDULER_TEST_LOG_LEVEL:-}" ]; then
    log_level="${SCHEDULER_TEST_LOG_LEVEL}"
  fi
  params="${params} ${log_level}"
  if [ -n "${SCHEDULER_TEST_ARGS:-}" ]; then
    params="${params} ${SCHEDULER_TEST_ARGS}"
  fi
  readonly kube_scheduler_docker_tag=$(cat "${kube_home}/kube-docker-files/kube-scheduler.docker_tag")

  # Remove salt comments and replace variables with values
  src_file="${kube_home}/kube-manifests/kubernetes/gci-trusty/kube-scheduler.manifest"
  remove_salt_config_comments "${src_file}"
  sed -i -e "s@{{params}}@${params}@g" "${src_file}"
  sed -i -e "s@{{pillar\['kube_docker_registry'\]}}@${DOCKER_REGISTRY}@g" "${src_file}"
  sed -i -e "s@{{pillar\['kube-scheduler_docker_tag'\]}}@${kube_scheduler_docker_tag}@g" "${src_file}"
  cp "${src_file}" /etc/kubernetes/manifests
}

# Starts a fluentd static pod for logging.
start_fluentd() {
  if [ "${ENABLE_NODE_LOGGING:-}" = "true" ]; then
    if [ "${LOGGING_DESTINATION:-}" = "gcp" ]; then
      cp /home/kubernetes/kube-manifests/kubernetes/fluentd-gcp.yaml /etc/kubernetes/manifests/
    elif [ "${LOGGING_DESTINATION:-}" = "elasticsearch" ]; then
      cp /home/kubernetes/kube-manifests/kubernetes/fluentd-es.yaml /etc/kubernetes/manifests/
    fi
  fi
}

# A helper function for copying addon manifests and set dir/files
# permissions.
# $1: addon category under /etc/kubernetes
# $2: manifest source dir
setup_addon_manifests() {
  src_dir="/home/kubernetes/kube-manifests/kubernetes/gci-trusty/$2"
  dst_dir="/etc/kubernetes/$1/$2"
  if [ ! -d "${dst_dir}" ]; then
    mkdir -p "${dst_dir}"
  fi
  files=$(find "${src_dir}" -name "*.yaml")
  if [ -n "${files}" ]; then
    cp "${src_dir}/"*.yaml "${dst_dir}"
  fi
  files=$(find "${src_dir}" -name "*.json")
  if [ -n "${files}" ]; then
    cp "${src_dir}/"*.json "${dst_dir}"
  fi
  files=$(find "${src_dir}" -name "*.yaml.in")
  if [ -n "${files}" ]; then
    cp "${src_dir}/"*.yaml.in "${dst_dir}"
  fi
  chown -R root:root "${dst_dir}"
  chmod 755 "${dst_dir}"
  chmod 644 "${dst_dir}"/*
}

# Prepares the manifests of k8s addons static pods.
prepare_kube_addons() {
  addon_src_dir="/home/kubernetes/kube-manifests/kubernetes/gci-trusty"
  addon_dst_dir="/etc/kubernetes/addons"
  # Set up manifests of other addons.
  if [ "${ENABLE_CLUSTER_MONITORING:-}" = "influxdb" ] || \
     [ "${ENABLE_CLUSTER_MONITORING:-}" = "google" ] || \
     [ "${ENABLE_CLUSTER_MONITORING:-}" = "standalone" ] || \
     [ "${ENABLE_CLUSTER_MONITORING:-}" = "googleinfluxdb" ]; then
    file_dir="cluster-monitoring/${ENABLE_CLUSTER_MONITORING}"
    setup_addon_manifests "addons" "${file_dir}"
    # Replace the salt configurations with variable values.
    metrics_memory="200Mi"
    eventer_memory="200Mi"
    readonly metrics_memory_per_node="4"
    readonly eventer_memory_per_node="500"
    if [ -n "${NUM_NODES:-}" ] && [ "${NUM_NODES}" -ge 1 ]; then
      num_kube_nodes="$((${NUM_NODES}-1))"
      metrics_memory="$((${num_kube_nodes} * ${metrics_memory_per_node} + 200))Mi"
      eventer_memory="$((${num_kube_nodes} * ${eventer_memory_per_node} + 200 * 1024))Ki"
    fi
    controller_yaml="${addon_dst_dir}/${file_dir}"
    if [ "${ENABLE_CLUSTER_MONITORING:-}" = "googleinfluxdb" ]; then
      controller_yaml="${controller_yaml}/heapster-controller-combined.yaml"
    else
      controller_yaml="${controller_yaml}/heapster-controller.yaml"
    fi
    remove_salt_config_comments "${controller_yaml}"
    sed -i -e "s@{{ *metrics_memory *}}@${metrics_memory}@g" "${controller_yaml}"
    sed -i -e "s@{{ *eventer_memory *}}@${eventer_memory}@g" "${controller_yaml}"
    sed -i -e "s@{{ *metrics_memory_per_node *}}@${metrics_memory_per_node}@g" "${controller_yaml}"
    sed -i -e "s@{{ *eventer_memory_per_node *}}@${eventer_memory_per_node}@g" "${controller_yaml}"
  fi
  cp "${addon_src_dir}/namespace.yaml" "${addon_dst_dir}"
  if [ "${ENABLE_L7_LOADBALANCING:-}" = "glbc" ]; then
    setup_addon_manifests "addons" "cluster-loadbalancing/glbc"
  fi
  if [ "${ENABLE_CLUSTER_DNS:-}" = "true" ]; then
    setup_addon_manifests "addons" "dns"
    dns_rc_file="${addon_dst_dir}/dns/skydns-rc.yaml"
    dns_svc_file="${addon_dst_dir}/dns/skydns-svc.yaml"
    mv "${addon_dst_dir}/dns/skydns-rc.yaml.in" "${dns_rc_file}"
    mv "${addon_dst_dir}/dns/skydns-svc.yaml.in" "${dns_svc_file}"
    # Replace the salt configurations with variable values.
    sed -i -e "s@{{ *pillar\['dns_replicas'\] *}}@${DNS_REPLICAS}@g" "${dns_rc_file}"
    sed -i -e "s@{{ *pillar\['dns_domain'\] *}}@${DNS_DOMAIN}@g" "${dns_rc_file}"
    sed -i -e "s@{{ *pillar\['dns_server'\] *}}@${DNS_SERVER_IP}@g" "${dns_svc_file}"
  fi
  if [ "${ENABLE_CLUSTER_REGISTRY:-}" = "true" ]; then
    setup_addon_manifests "addons" "registry"
    registry_pv_file="${addon_dst_dir}/registry/registry-pv.yaml"
    registry_pvc_file="${addon_dst_dir}/registry/registry-pvc.yaml"
    mv "${addon_dst_dir}/registry/registry-pv.yaml.in" "${registry_pv_file}"
    mv "${addon_dst_dir}/registry/registry-pvc.yaml.in" "${registry_pvc_file}"
    # Replace the salt configurations with variable values.
    remove_salt_config_comments "${controller_yaml}"
    sed -i -e "s@{{ *pillar\['cluster_registry_disk_size'\] *}}@${CLUSTER_REGISTRY_DISK_SIZE}@g" "${registry_pv_file}"
    sed -i -e "s@{{ *pillar\['cluster_registry_disk_size'\] *}}@${CLUSTER_REGISTRY_DISK_SIZE}@g" "${registry_pvc_file}"
    sed -i -e "s@{{ *pillar\['cluster_registry_disk_name'\] *}}@${CLUSTER_REGISTRY_DISK}@g" "${registry_pvc_file}"
  fi
  if [ "${ENABLE_NODE_LOGGING:-}" = "true" ] && \
     [ "${LOGGING_DESTINATION:-}" = "elasticsearch" ] && \
     [ "${ENABLE_CLUSTER_LOGGING:-}" = "true" ]; then
    setup_addon_manifests "addons" "fluentd-elasticsearch"
  fi
  if [ "${ENABLE_CLUSTER_UI:-}" = "true" ]; then
    setup_addon_manifests "addons" "dashboard"
  fi
  if echo "${ADMISSION_CONTROL:-}" | grep -q "LimitRanger"; then
    setup_addon_manifests "admission-controls" "limit-range"
  fi

  # Place addon manager pod manifest
  cp "${addon_src_dir}/kube-addon-manager.yaml" /etc/kubernetes/manifests
}
