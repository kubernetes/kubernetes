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
# master and nodes. It is uploaded as GCE instance metadata. The upstart jobs
# in cluster/gce/trusty/<node.yaml, master.yaml> download it and make use
# of needed functions. The script itself is not supposed to be executed in
# other manners.

download_kube_env() {
  # Fetch kube-env from GCE metadata server.
  readonly tmp_install_dir="/var/cache/kubernetes-install"
  mkdir -p "${tmp_install_dir}"
  curl --fail --silent --show-error \
    -H "X-Google-Metadata-Request: True" \
    -o "${tmp_install_dir}/kube_env.yaml" \
    http://metadata.google.internal/computeMetadata/v1/instance/attributes/kube-env
  # Convert the yaml format file into a shell-style file.
  eval $(python -c '''
import pipes,sys,yaml
for k,v in yaml.load(sys.stdin).iteritems():
  print("readonly {var}={value}".format(var = k, value = pipes.quote(str(v))))
''' < "${tmp_install_dir}/kube_env.yaml" > /etc/kube-env)
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
  # In anyway we have to download the release tarball as docker_tag files and
  # kube-proxy image file are there.
  cd /tmp
  k8s_sha1="${SERVER_BINARY_TAR_URL##*/}.sha1"
  echo "Downloading k8s tar sha1 file ${k8s_sha1}"
  download_or_bust "${k8s_sha1}" "${SERVER_BINARY_TAR_URL}.sha1"
  k8s_tar="${SERVER_BINARY_TAR_URL##*/}"
  echo "Downloading k8s tar file ${k8s_tar}"
  download_or_bust "${k8s_tar}" "${SERVER_BINARY_TAR_URL}"
  # Validate hash.
  actual=$(sha1sum "${k8s_tar}" | awk '{ print $1 }') || true
  if [ "${actual}" != "${SERVER_BINARY_TAR_HASH}" ]; then
    echo "== ${k8s_tar} corrupted, sha1 ${actual} doesn't match expected ${SERVER_BINARY_TAR_HASH} =="
  else
    echo "Validated ${SERVER_BINARY_TAR_URL} SHA1 = ${SERVER_BINARY_TAR_HASH}"
  fi
  tar xzf "/tmp/${k8s_tar}" -C /tmp/ --overwrite
  # Copy docker_tag and image files to /run/kube-docker-files.
  mkdir -p /run/kube-docker-files
  cp /tmp/kubernetes/server/bin/*.docker_tag /run/kube-docker-files/
  if [ "${KUBERNETES_MASTER:-}" = "false" ]; then
    cp /tmp/kubernetes/server/bin/kube-proxy.tar /run/kube-docker-files/
  else
    cp /tmp/kubernetes/server/bin/kube-apiserver.tar /run/kube-docker-files/
    cp /tmp/kubernetes/server/bin/kube-controller-manager.tar /run/kube-docker-files/
    cp /tmp/kubernetes/server/bin/kube-scheduler.tar /run/kube-docker-files/
    cp -r /tmp/kubernetes/addons /run/kube-docker-files/
  fi
  # Use the binary from the release tarball if they are not preinstalled, or if this is
  # a test cluster.
  readonly BIN_PATH="/usr/bin"
  if ! which kubelet > /dev/null || ! which kubectl > /dev/null; then
    cp /tmp/kubernetes/server/bin/kubelet "${BIN_PATH}"
    cp /tmp/kubernetes/server/bin/kubectl "${BIN_PATH}"
  elif [ "${TEST_CLUSTER:-}" = "true" ]; then
    mkdir -p /home/kubernetes/bin
    cp /tmp/kubernetes/server/bin/kubelet /home/kubernetes/bin
    cp /tmp/kubernetes/server/bin/kubectl /home/kubernetes/bin
    mount --bind /home/kubernetes/bin/kubelet "${BIN_PATH}/kubelet"
    mount --bind -o remount,ro,^noexec "${BIN_PATH}/kubelet" "${BIN_PATH}/kubelet"
    mount --bind /home/kubernetes/bin/kubectl "${BIN_PATH}/kubectl"
    mount --bind -o remount,ro,^noexec "${BIN_PATH}/kubectl" "${BIN_PATH}/kubectl"
  fi
  # Clean up.
  rm -rf /tmp/kubernetes
  rm "/tmp/${k8s_tar}"
  rm "/tmp/${k8s_sha1}"

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
  actual=$(sha1sum "${manifests_tar}" | awk '{ print $1 }') || true
  if [ "${actual}" != "${KUBE_MANIFESTS_TAR_HASH}" ]; then
    echo "== ${manifests_tar} corrupted, sha1 ${actual} doesn't match expected ${KUBE_MANIFESTS_TAR_HASH} =="
  else
    echo "Validated ${KUBE_MANIFESTS_TAR_URL} SHA1 = ${KUBE_MANIFESTS_TAR_HASH}"
  fi
  tar xzf "/run/kube-manifests/${manifests_tar}" -C /run/kube-manifests/ --overwrite
  readonly kube_addon_registry="${KUBE_ADDON_REGISTRY:-gcr.io/google_containers}"
  if [ "${kube_addon_registry}" != "gcr.io/google_containers" ]; then
    find /run/kube-manifests -name \*.yaml -or -name \*.yaml.in | \
      xargs sed -ri "s@(image:\s.*)gcr.io/google_containers@\1${kube_addon_registry}@"
    find /run/kube-manifests -name \*.manifest -or -name \*.json | \
      xargs sed -ri "s@(image\":\s+\")gcr.io/google_containers@\1${kube_addon_registry}@"
  fi
  cp /run/kube-manifests/kubernetes/trusty/configure-helper.sh /etc/kube-configure-helper.sh
  rm "/run/kube-manifests/${manifests_sha1}"
  rm "/run/kube-manifests/${manifests_tar}"
}
