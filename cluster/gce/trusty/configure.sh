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
# nodes. It is uploaded to GCE metadata server when a VM instance is created,
# and then downloaded by the instance. The upstart jobs in
# cluster/gce/trusty/node.yaml source this script to make use of needed
# functions. The script itself is not supposed to be executed in other manners.

config_hostname() {
  # Set the hostname to the short version.
  short_hostname=$(hostname -s)
  hostname $short_hostname
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
  mkdir -p /var/lib/kube-proxy
  mkdir -p /etc/kubernetes/manifests
}

download_kube_env() {
  # Fetch kube-env from GCE metadata server.
  curl --fail --silent --show-error \
    -H "X-Google-Metadata-Request: True" \
    -o /tmp/kube-env-yaml \
    http://metadata.google.internal/computeMetadata/v1/instance/attributes/kube-env
  # Convert the yaml format file into a shell-style file.
  eval $(python -c '''
import pipes,sys,yaml
for k,v in yaml.load(sys.stdin).iteritems():
  print "readonly {var}={value}".format(var = k, value = pipes.quote(str(v)))
''' < /tmp/kube-env-yaml > /etc/kube-env)
}

create_kubelet_kubeconfig() {
  # Create the kubelet kubeconfig file.
  . /etc/kube-env
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
    # We should install the latest qualified docker, which is version 1.8.3 at present.
    curl -sSL https://get.docker.com/ | DOCKER_VERSION=1.8.3 sh
  fi
  if ! which brctl > /dev/null; then
    echo "Do not find brctl. Install it."
    apt-get install --yes bridge-utils
  fi
}

# Install the packages that are useful but not required by spinning up a cluster.
install_additional_packages() {
  # Socat and nsenter are not required for spinning up a cluster. We move the
  # installation here to be in parallel with the cluster creation.
  if ! which socat > /dev/null; then
    echo "Do not find socat. Install it."
    apt-get install --yes socat
  fi
  if ! which nsenter > /dev/null; then
    echo "Do not find nsenter. Install it."
    # Note: this is an easy way to install nsenter, but may not be the fastest
    # way. In addition, this may not be a trusted source. So, replace it if
    # we have a better solution.
    docker run --rm -v /usr/local/bin:/target jpetazzo/nsenter
  fi
}

# Retry a download until we get it.
#
# $1 is the file to create
# $2 is the URL to download
download_or_bust() {
  rm -f $1 > /dev/null
  until curl --ipv4 -Lo "$1" --connect-timeout 20 --retry 6 --retry-delay 10 "$2"; do
    echo "Failed to download file ($2). Retrying."
  done
}

# Downloads kubernetes binaries and kube-system manifest tarball, unpacks them,
# and places them into suitable directories.
install_kube_binary_config() {
  . /etc/kube-env
  # For a testing cluster, we pull kubelet, kube-proxy, and kubectl binaries,
  # and place them in /usr/local/bin. For a non-test cluster, we use the binaries
  # pre-installed in the image, or pull and place them in /usr/bin if they are
  # not pre-installed.
  BINARY_PATH="/usr/bin/"
  if [ "${TEST_CLUSTER:-}" = "true" ]; then
    BINARY_PATH="/usr/local/bin/"
  fi
  if ! which kubelet > /dev/null || ! which kube-proxy > /dev/null || [ "${TEST_CLUSTER:-}" = "true" ]; then
    cd /tmp
    k8s_sha1="${SERVER_BINARY_TAR_URL##*/}.sha1"
    echo "Downloading k8s tar sha1 file ${k8s_sha1}"
    download_or_bust "${k8s_sha1}" "${SERVER_BINARY_TAR_URL}.sha1"
    k8s_tar="${SERVER_BINARY_TAR_URL##*/}"
    echo "Downloading k8s tar file ${k8s_tar}"
    download_or_bust "${k8s_tar}" "${SERVER_BINARY_TAR_URL}"
    # Validate hash.
    actual=$(sha1sum ${k8s_tar} | awk '{ print $1 }') || true
    if [ "${actual}" != "${SERVER_BINARY_TAR_HASH}" ]; then
      echo "== ${k8s_tar} corrupted, sha1 ${actual} doesn't match expected ${SERVER_BINARY_TAR_HASH} =="
    else
      echo "Validated ${SERVER_BINARY_TAR_URL} SHA1 = ${SERVER_BINARY_TAR_HASH}"
    fi
    tar xzf "/tmp/${k8s_tar}" -C /tmp/ --overwrite
    cp /tmp/kubernetes/server/bin/kubelet ${BINARY_PATH}
    cp /tmp/kubernetes/server/bin/kube-proxy ${BINARY_PATH}
    cp /tmp/kubernetes/server/bin/kubectl ${BINARY_PATH}
    rm -rf "/tmp/kubernetes"
    rm "/tmp/${k8s_tar}"
    rm "/tmp/${k8s_sha1}"
  fi

  # Put kube-system pods manifests in /etc/kube-manifests/.
  mkdir -p /run/kube-manifests
  cd /run/kube-manifests
  manifests_sha1="${KUBE_MANIFESTS_TAR_URL##*/}.sha1"
  echo "Downloading kube-system manifests tar sha1 file ${manifests_sha1}"
  download_or_bust "${manifests_sha1}" "${KUBE_MANIFESTS_TAR_URL}.sha1"
  manifests_tar="${KUBE_MANIFESTS_TAR_URL##*/}"
  echo "Downloading kube-manifest tar file ${manifests_tar}"
  download_or_bust "${manifests_tar}" "${KUBE_MANIFESTS_TAR_URL}"
  # Validate hash.
  actual=$(sha1sum ${manifests_tar} | awk '{ print $1 }') || true
  if [ "${actual}" != "${KUBE_MANIFESTS_TAR_HASH}" ]; then
    echo "== ${manifests_tar} corrupted, sha1 ${actual} doesn't match expected ${KUBE_MANIFESTS_TAR_HASH} =="
  else
    echo "Validated ${KUBE_MANIFESTS_TAR_URL} SHA1 = ${KUBE_MANIFESTS_TAR_HASH}"
  fi
  tar xzf "/run/kube-manifests/${manifests_tar}" -C /run/kube-manifests/ --overwrite
  rm "/run/kube-manifests/${manifests_sha1}"
  rm "/run/kube-manifests/${manifests_tar}"
}

restart_docker_daemon() {
  . /etc/kube-env
  # Assemble docker deamon options
  DOCKER_OPTS="-p /var/run/docker.pid --bridge=cbr0 --iptables=false --ip-masq=false"
  if [ "${TEST_CLUSTER:-}" = "true" ]; then
    DOCKER_OPTS="${DOCKER_OPTS} --log-level=debug"
  fi
  echo "DOCKER_OPTS=\"${DOCKER_OPTS} ${EXTRA_DOCKER_OPTS:-}\"" > /etc/default/docker
  # Make sure the network interface cbr0 is created before restarting docker daemon
  while ! [ -L /sys/class/net/cbr0 ]; do
    echo "Sleep 1 second to wait for cbr0"
    sleep 1
  done
  initctl restart docker
  # Remove docker0
  ifconfig docker0 down
  brctl delbr docker0
}
