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

# This script contains functions for configuring instances to run kubernetes
# master and nodes. It is uploaded as GCE instance metadata. The upstart jobs
# in cluster/gce/trusty/<node.yaml, master.yaml> download it and make use
# of needed functions. The script itself is not supposed to be executed in
# other manners.

set_broken_motd() {
  cat > /etc/motd <<EOF
Broken (or in progress) Kubernetes node setup! If you are using Ubuntu Trusty,
check log file /var/log/syslog. If you are using GCI image, use
"journalctl | grep kube" to find more information.
EOF
}

download_kube_env() {
  # Fetch kube-env from GCE metadata server.
  readonly tmp_kube_env="/tmp/kube-env.yaml"
  curl --fail --retry 5 --retry-delay 3 --silent --show-error \
    -H "X-Google-Metadata-Request: True" \
    -o "${tmp_kube_env}" \
    http://metadata.google.internal/computeMetadata/v1/instance/attributes/kube-env
  # Convert the yaml format file into a shell-style file.
  eval $(python -c '''
import pipes,sys,yaml
for k,v in yaml.load(sys.stdin).iteritems():
  print("readonly {var}={value}".format(var = k, value = pipes.quote(str(v))))
''' < "${tmp_kube_env}" > /etc/kube-env)
  rm -f "${tmp_kube_env}"
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
# and places them into suitable directories. Files are placed in /home/kubernetes. 
install_kube_binary_config() {
  # Upstart does not support shell array well. Put urls in a temp file with one
  # url at a line, and we will use 'read' command to get them one-by-one.
  tmp_binary_urls=$(mktemp /tmp/kube-temp.XXXXXX)
  echo "${SERVER_BINARY_TAR_URL}" | tr "," "\n" > "${tmp_binary_urls}"
  tmp_manifests_urls=$(mktemp /tmp/kube-temp.XXXXXX)
  echo "${KUBE_MANIFESTS_TAR_URL}" | tr "," "\n" > "${tmp_manifests_urls}"

  kube_home="/home/kubernetes"
  mkdir -p "${kube_home}"
  cd "${kube_home}"
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
  tar xzf "${kube_home}/${server_binary_tar}" -C "${kube_home}" --overwrite
  # Copy docker_tag and image files to /home/kubernetes/kube-docker-files.
  src_dir="${kube_home}/kubernetes/server/bin"
  dst_dir="${kube_home}/kube-docker-files"
  mkdir -p "${dst_dir}"
  cp "${src_dir}/"*.docker_tag "${dst_dir}"
  if [ "${KUBERNETES_MASTER:-}" = "false" ]; then
    cp "${src_dir}/kube-proxy.tar" "${dst_dir}"
  else
    cp "${src_dir}/kube-apiserver.tar" "${dst_dir}"
    cp "${src_dir}/kube-controller-manager.tar" "${dst_dir}"
    cp "${src_dir}/kube-scheduler.tar" "${dst_dir}"
    cp -r "${kube_home}/kubernetes/addons" "${dst_dir}"
  fi
  # Use the binary from the release tarball if they are not preinstalled, or if this is
  # a test cluster.
  readonly BIN_PATH="/usr/bin"
  if ! which kubelet > /dev/null || ! which kubectl > /dev/null; then
    # This should be the case of trusty.
    cp "${src_dir}/kubelet" "${BIN_PATH}"
    cp "${src_dir}/kubectl" "${BIN_PATH}"
  else
    # This should be the case of GCI.
    readonly kube_bin="${kube_home}/bin"
    mkdir -p "${kube_bin}"
    mount --bind "${kube_bin}" "${kube_bin}"
    mount -o remount,rw,exec "${kube_bin}"
    cp "${src_dir}/kubelet" "${kube_bin}"
    cp "${src_dir}/kubectl" "${kube_bin}"
    chmod 544 "${kube_bin}/kubelet"
    chmod 544 "${kube_bin}/kubectl"
    # If the built-in binary version is different from the expected version, we use
    # the downloaded binary. The simplest implementation is to always use the downloaded
    # binary without checking the version. But we have another version guardian in GKE.
    # So, we compare the versions to ensure this run-time binary replacement is only
    # applied for OSS kubernetes.
    readonly builtin_version="$(/usr/bin/kubelet --version=true | cut -f2 -d " ")"
    readonly required_version="$(/home/kubernetes/bin/kubelet --version=true | cut -f2 -d " ")"
    if [ "${TEST_CLUSTER:-}" = "true" ] || [ "${builtin_version}" != "${required_version}" ]; then
      mount --bind "${kube_bin}/kubelet" "${BIN_PATH}/kubelet"
      mount --bind "${kube_bin}/kubectl" "${BIN_PATH}/kubectl"
    else
      # Remove downloaded binary just to prevent misuse.
      rm -f "${kube_bin}/kubelet"
      rm -f "${kube_bin}/kubectl"
    fi
  fi
  cp "${kube_home}/kubernetes/LICENSES" "${kube_home}"

  # Put kube-system pods manifests in /home/kubernetes/kube-manifests/.
  dst_dir="${kube_home}/kube-manifests"
  mkdir -p "${dst_dir}"
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
  tar xzf "${kube_home}/${manifests_tar}" -C "${dst_dir}" --overwrite
  readonly kube_addon_registry="${KUBE_ADDON_REGISTRY:-gcr.io/google_containers}"
  if [ "${kube_addon_registry}" != "gcr.io/google_containers" ]; then
    find "${dst_dir}" -name \*.yaml -or -name \*.yaml.in | \
      xargs sed -ri "s@(image:\s.*)gcr.io/google_containers@\1${kube_addon_registry}@"
    find "${dst_dir}" -name \*.manifest -or -name \*.json | \
      xargs sed -ri "s@(image\":\s+\")gcr.io/google_containers@\1${kube_addon_registry}@"
  fi
  cp "${dst_dir}/kubernetes/gci-trusty/trusty-configure-helper.sh" /etc/kube-configure-helper.sh

  # Clean up.
  rm -rf "${kube_home}/kubernetes"
  rm -f "${kube_home}/${server_binary_tar}"
  rm -f "${kube_home}/${server_binary_tar}.sha1"
  rm -f "${kube_home}/${manifests_tar}"
  rm -f "${kube_home}/${manifests_tar}.sha1"
  rm -f "${tmp_binary_urls}"
  rm -f "${tmp_manifests_urls}"
}
