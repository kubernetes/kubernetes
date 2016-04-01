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

validate_hash() {
  file="$1"
  expected="$2"

  actual=$(sha1sum ${file} | awk '{ print $1 }') || true
  if [ "${actual}" != "${expected}" ]; then
    echo "== ${file} corrupted, sha1 ${actual} doesn't match expected ${expected} =="
    return 1
  fi
}

# Retry a download until we get it. Takes a hash and a set of URLs.
#
# $1: The sha1 of the URL. Can be "" if the sha1 is unknown, which means
#     we are downloading a hash file.
# $2: The temp file containing a list of urls to download.
download_or_bust() {
  file_hash="$1"
  tmpfile_urls="$2"

  while true; do
    # Read urls from the file one-by-one.
    while read -r url; do
      if [ ! -n "${file_hash}" ]; then
        url="${url/.tar.gz/.tar.gz.sha1}"
      fi
      file="${url##*/}"
      rm -f "${file}"
      if ! curl -f --ipv4 -Lo "${file}" --connect-timeout 20 --retry 6 --retry-delay 10 "${url}"; then
        echo "== Failed to download ${url}. Retrying. =="
      elif [ -n "${file_hash}" ] && ! validate_hash "${file}" "${file_hash}"; then
        echo "== Hash validation of ${url} failed. Retrying. =="
      else
        if [ -n "${file_hash}" ]; then
          echo "== Downloaded ${url} (SHA1 = ${file_hash}) =="
        else
          echo "== Downloaded ${url} =="
        fi
        return
      fi
    done < "${tmpfile_urls}"
  done
}

# Downloads kubernetes binaries and kube-system manifest tarball, unpacks them,
# and places them into suitable directories.
install_kube_binary_config() {
  # Upstart does not support shell array well. Put urls in a temp file with one
  # url at a line, and we will use 'read' command to get them one-by-one.
  tmp_binary_urls=$(mktemp /tmp/kube-temp.XXXXXX)
  echo "${SERVER_BINARY_TAR_URL}" | tr "," "\n" > "${tmp_binary_urls}"
  tmp_manifests_urls=$(mktemp /tmp/kube-temp.XXXXXX)
  echo "${KUBE_MANIFESTS_TAR_URL}" | tr "," "\n" > "${tmp_manifests_urls}"

  cd /tmp
  read -r server_binary_tar_url < "${tmp_binary_urls}"
  readonly server_binary_tar="${server_binary_tar_url##*/}"
  if [ -n "${SERVER_BINARY_TAR_HASH:-}" ]; then
    readonly server_binary_tar_hash="${SERVER_BINARY_TAR_HASH}"
  else
    echo "Downloading binary release sha1 (not found in env)"
    download_or_bust "" "${tmp_binary_urls}"
    readonly server_binary_tar_hash=$(cat "${server_binary_tar}.sha1")
  fi
  echo "Downloading binary release tar"
  download_or_bust "${server_binary_tar_hash}" "${tmp_binary_urls}"
  tar xzf "/tmp/${server_binary_tar}" -C /tmp/ --overwrite
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

  # Put kube-system pods manifests in /etc/kube-manifests/.
  mkdir -p /run/kube-manifests
  cd /run/kube-manifests
  read -r manifests_tar_url < "${tmp_manifests_urls}"
  readonly manifests_tar="${manifests_tar_url##*/}"
  if [ -n "${KUBE_MANIFESTS_TAR_HASH:-}" ]; then
    readonly manifests_tar_hash="${KUBE_MANIFESTS_TAR_HASH}"
  else
    echo "Downloading k8s manifests sha1 (not found in env)"
    download_or_bust "" "${tmp_manifests_urls}"
    readonly manifests_tar_hash=$(cat "${manifests_tar}.sha1")
  fi
  echo "Downloading k8s manifests tar"
  download_or_bust "${manifests_tar_hash}" "${tmp_manifests_urls}"
  tar xzf "/run/kube-manifests/${manifests_tar}" -C /run/kube-manifests/ --overwrite
  readonly kube_addon_registry="${KUBE_ADDON_REGISTRY:-gcr.io/google_containers}"
  if [ "${kube_addon_registry}" != "gcr.io/google_containers" ]; then
    find /run/kube-manifests -name \*.yaml -or -name \*.yaml.in | \
      xargs sed -ri "s@(image:\s.*)gcr.io/google_containers@\1${kube_addon_registry}@"
    find /run/kube-manifests -name \*.manifest -or -name \*.json | \
      xargs sed -ri "s@(image\":\s+\")gcr.io/google_containers@\1${kube_addon_registry}@"
  fi
  cp /run/kube-manifests/kubernetes/trusty/configure-helper.sh /etc/kube-configure-helper.sh

  # Clean up.
  rm -rf /tmp/kubernetes
  rm -f "/tmp/${server_binary_tar}"
  rm -f "/tmp/${server_binary_tar}.sha1"
  rm -f "/run/kube-manifests/${manifests_tar}"
  rm -f "/run/kube-manifests/${manifests_tar}.sha1"
  rm -f "${tmp_binary_urls}"
  rm -f "${tmp_manifests_urls}"
}
