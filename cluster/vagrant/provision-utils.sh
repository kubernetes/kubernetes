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

function write-salt-config() {
  local role="$1"

  # Update salt configuration
  mkdir -p /etc/salt/minion.d

  mkdir -p /srv/salt-overlay/pillar
  cat <<EOF >/srv/salt-overlay/pillar/cluster-params.sls
service_cluster_ip_range: '$(echo "$SERVICE_CLUSTER_IP_RANGE" | sed -e "s/'/''/g")'
cert_ip: '$(echo "$MASTER_IP" | sed -e "s/'/''/g")'
enable_cluster_monitoring: '$(echo "$ENABLE_CLUSTER_MONITORING" | sed -e "s/'/''/g")'
enable_cluster_logging: '$(echo "$ENABLE_CLUSTER_LOGGING" | sed -e "s/'/''/g")'
enable_cluster_ui: '$(echo "$ENABLE_CLUSTER_UI" | sed -e "s/'/''/g")'
enable_node_logging: '$(echo "$ENABLE_NODE_LOGGING" | sed -e "s/'/''/g")'
logging_destination: '$(echo "$LOGGING_DESTINATION" | sed -e "s/'/''/g")'
elasticsearch_replicas: '$(echo "$ELASTICSEARCH_LOGGING_REPLICAS" | sed -e "s/'/''/g")'
enable_cluster_dns: '$(echo "$ENABLE_CLUSTER_DNS" | sed -e "s/'/''/g")'
dns_replicas: '$(echo "$DNS_REPLICAS" | sed -e "s/'/''/g")'
dns_server: '$(echo "$DNS_SERVER_IP" | sed -e "s/'/''/g")'
dns_domain: '$(echo "$DNS_DOMAIN" | sed -e "s/'/''/g")'
instance_prefix: '$(echo "$INSTANCE_PREFIX" | sed -e "s/'/''/g")'
admission_control: '$(echo "$ADMISSION_CONTROL" | sed -e "s/'/''/g")'
enable_cpu_cfs_quota: '$(echo "$ENABLE_CPU_CFS_QUOTA" | sed -e "s/'/''/g")'
network_provider: '$(echo "$NETWORK_PROVIDER" | sed -e "s/'/''/g")'
opencontrail_tag: '$(echo "$OPENCONTRAIL_TAG" | sed -e "s/'/''/g")'
opencontrail_kubernetes_tag: '$(echo "$OPENCONTRAIL_KUBERNETES_TAG" | sed -e "s/'/''/g")'
opencontrail_public_subnet: '$(echo "$OPENCONTRAIL_PUBLIC_SUBNET" | sed -e "s/'/''/g")'
e2e_storage_test_environment: '$(echo "$E2E_STORAGE_TEST_ENVIRONMENT" | sed -e "s/'/''/g")'
EOF

  cat <<EOF >/etc/salt/minion.d/log-level-debug.conf
log_level: warning
log_level_logfile: warning
EOF

  cat <<EOF >/etc/salt/minion.d/grains.conf
grains:
  node_ip: '$(echo "$MASTER_IP" | sed -e "s/'/''/g")'
  publicAddressOverride: '$(echo "$MASTER_IP" | sed -e "s/'/''/g")'
  network_mode: openvswitch
  networkInterfaceName: '$(echo "$NETWORK_IF_NAME" | sed -e "s/'/''/g")'
  api_servers: '$(echo "$MASTER_IP" | sed -e "s/'/''/g")'
  cloud: vagrant
  roles:
    - $role
  runtime_config: '$(echo "$RUNTIME_CONFIG" | sed -e "s/'/''/g")'
  docker_opts: '$(echo "$DOCKER_OPTS" | sed -e "s/'/''/g")'
  master_extra_sans: '$(echo "$MASTER_EXTRA_SANS" | sed -e "s/'/''/g")'
  keep_host_etcd: true
EOF
}

function release_not_found() {
    echo "It looks as if you don't have a compiled version of Kubernetes.  If you" >&2
    echo "are running from a clone of the git repo, please run 'make quick-release'." >&2
    echo "Note that this requires having Docker installed.  If you are running " >&2
    echo "from a release tarball, something is wrong.  Look at " >&2
    echo "http://kubernetes.io/ for information on how to contact the development team for help." >&2
    exit 1
}

function install-salt() {
  server_binary_tar="/vagrant/server/kubernetes-server-linux-amd64.tar.gz"
  if [[ ! -f "$server_binary_tar" ]]; then
    server_binary_tar="/vagrant/_output/release-tars/kubernetes-server-linux-amd64.tar.gz"
  fi
  if [[ ! -f "$server_binary_tar" ]]; then
    release_not_found
  fi

  salt_tar="/vagrant/server/kubernetes-salt.tar.gz"
  if [[ ! -f "$salt_tar" ]]; then
    salt_tar="/vagrant/_output/release-tars/kubernetes-salt.tar.gz"
  fi
  if [[ ! -f "$salt_tar" ]]; then
    release_not_found
  fi

  echo "Running release install script"
  rm -rf /kube-install
  mkdir -p /kube-install
  pushd /kube-install
  tar xzf "$salt_tar"
  cp "$server_binary_tar" .
  ./kubernetes/saltbase/install.sh "${server_binary_tar##*/}"
  popd

  if ! which salt-call >/dev/null 2>&1; then
    # Install salt binaries
    curl -sS -L --connect-timeout 20 --retry 6 --retry-delay 10 https://bootstrap.saltstack.com | sh -s
  fi
}

function run-salt() {
  echo "  Now waiting for the Salt provisioning process to complete on this machine."
  echo "  This can take some time based on your network, disk, and cpu speed."
  salt-call --local state.highstate
}

function create-salt-kubelet-auth() {
  local -r kubelet_kubeconfig_folder="/srv/salt-overlay/salt/kubelet"
  mkdir -p "${kubelet_kubeconfig_folder}"
  (umask 077;
  cat > "${kubelet_kubeconfig_folder}/kubeconfig" << EOF
apiVersion: v1
kind: Config
clusters:
- cluster:
    insecure-skip-tls-verify: true
  name: local
contexts:
- context:
    cluster: local
    user: kubelet
  name: service-account-context
current-context: service-account-context
users:
- name: kubelet
  user:
    token: ${KUBELET_TOKEN}
EOF
  )
}

function create-salt-kubeproxy-auth() {
  kube_proxy_kubeconfig_folder="/srv/salt-overlay/salt/kube-proxy"
  mkdir -p "${kube_proxy_kubeconfig_folder}"
  (umask 077;
  cat > "${kube_proxy_kubeconfig_folder}/kubeconfig" << EOF
apiVersion: v1
kind: Config
clusters:
- cluster:
    insecure-skip-tls-verify: true
  name: local
contexts:
- context:
    cluster: local
    user: kube-proxy
  name: service-account-context
current-context: service-account-context
users:
- name: kube-proxy
  user:
    token: ${KUBE_PROXY_TOKEN}
EOF
  )
}

# provision-network runs some generic node network provisioning logic
function provision-network() {
  # Set the host name explicitly
  # See: https://github.com/mitchellh/vagrant/issues/2430
  hostnamectl set-hostname ${MASTER_NAME}

  if [[ "$(grep 'VERSION_ID' /etc/os-release)" =~ ^VERSION_ID=21 ]]; then
    # Workaround to vagrant inability to guess interface naming sequence
    # Tell system to abandon the new naming scheme and use eth* instead
    rm -f /etc/sysconfig/network-scripts/ifcfg-enp0s3

    # Disable network interface being managed by Network Manager (needed for Fedora 21+)
    NETWORK_CONF_PATH=/etc/sysconfig/network-scripts/
    if_to_edit=$( find ${NETWORK_CONF_PATH}ifcfg-* | xargs grep -l VAGRANT-BEGIN )
    for if_conf in ${if_to_edit}; do
      grep -q ^NM_CONTROLLED= ${if_conf} || echo 'NM_CONTROLLED=no' >> ${if_conf}
      sed -i 's/#^NM_CONTROLLED=.*/NM_CONTROLLED=no/' ${if_conf}
    done;
    systemctl restart network
  fi

  NETWORK_IF_NAME=`echo ${if_to_edit} | awk -F- '{ print $3 }'`

  # Setup hosts file to support ping by hostname to each minion in the cluster from apiserver
  for (( i=0; i<${#NODE_NAMES[@]}; i++)); do
    minion=${NODE_NAMES[$i]}
    ip=${NODE_IPS[$i]}
    if [ ! "$(cat /etc/hosts | grep $minion)" ]; then
      echo "Adding $minion to hosts file"
      echo "$ip $minion" >> /etc/hosts
    fi
  done

  echo "127.0.0.1 localhost" >> /etc/hosts
  echo "$MASTER_IP $MASTER_NAME" >> /etc/hosts
}

# provision-flannel-node configures a flannel node
function provision-flannel-node {

  echo "Provisioning flannel on node"

  FLANNEL_ETCD_URL="http://${MASTER_IP}:4379"

  # Install flannel for overlay
  if ! which flanneld >/dev/null 2>&1; then
    yum install -y flannel

    # Configure local daemon to speak to master
    NETWORK_CONF_PATH=/etc/sysconfig/network-scripts/
    if_to_edit=$( find ${NETWORK_CONF_PATH}ifcfg-* | xargs grep -l VAGRANT-BEGIN )
    NETWORK_IF_NAME=`echo ${if_to_edit} | awk -F- '{ print $3 }'`
    cat <<EOF >/etc/sysconfig/flanneld
FLANNEL_ETCD="${FLANNEL_ETCD_URL}"
FLANNEL_ETCD_KEY="/coreos.com/network"
FLANNEL_OPTIONS="-iface=${NETWORK_IF_NAME}"
EOF

    # Start flannel
    systemctl enable flanneld
    systemctl start flanneld
  fi

  echo "Network configuration verified"
}

# provision-flannel-master configures flannel on the master
function provision-flannel-master {

  echo "Provisioning flannel master configuration"

  FLANNEL_ETCD_URL="http://${MASTER_IP}:4379"

  # Install etcd for flannel data
  if ! which etcd >/dev/null 2>&1; then

    yum install -y etcd

    # Modify etcd configuration for flannel data
    cat <<EOF >/etc/etcd/etcd.conf
ETCD_NAME=flannel
ETCD_DATA_DIR="/var/lib/etcd/flannel.etcd"
ETCD_LISTEN_PEER_URLS="http://${MASTER_IP}:4380"
ETCD_LISTEN_CLIENT_URLS="http://${MASTER_IP}:4379"
ETCD_INITIAL_ADVERTISE_PEER_URLS="http://${MASTER_IP}:4380"
ETCD_INITIAL_CLUSTER="flannel=http://${MASTER_IP}:4380"
ETCD_ADVERTISE_CLIENT_URLS="${FLANNEL_ETCD_URL}"
EOF
    # Enable and start etcd
    systemctl enable etcd
    systemctl start etcd

  fi

  cat <<EOF >/etc/flannel-config.json
{
    "Network": "${CONTAINER_SUBNET}",
    "SubnetLen": 24,
    "Backend": {
        "Type": "udp",
        "Port": 8285
     }
}
EOF

  # Import default configuration into etcd for master setup
  etcdctl -C ${FLANNEL_ETCD_URL} set /coreos.com/network/config < /etc/flannel-config.json

  echo "Network configuration verified"
}
