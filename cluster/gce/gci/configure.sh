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

# Due to the GCE custom metadata size limit, we split the entire script into two
# files configure.sh and configure-helper.sh. The functionality of downloading
# kubernetes configuration, manifests, docker images, and binary files are
# put in configure.sh, which is uploaded via GCE custom metadata.

set -o errexit
set -o nounset
set -o pipefail

function set-broken-motd {
  cat > /etc/motd <<EOF
Broken (or in progress) Kubernetes node setup! Check the cluster initialization status
using the following commands.

Master instance:
  - sudo systemctl status kube-master-installation
  - sudo systemctl status kube-master-configuration

Node instance:
  - sudo systemctl status kube-node-installation
  - sudo systemctl status kube-node-configuration
EOF
}

function download-kube-env {
  # Fetch kube-env from GCE metadata server.
  local -r tmp_kube_env="/tmp/kube-env.yaml"
  curl --fail --retry 5 --retry-delay 3 --silent --show-error \
    -H "X-Google-Metadata-Request: True" \
    -o "${tmp_kube_env}" \
    http://metadata.google.internal/computeMetadata/v1/instance/attributes/kube-env
  # Convert the yaml format file into a shell-style file.
  eval $(python -c '''
import pipes,sys,yaml
for k,v in yaml.load(sys.stdin).iteritems():
  print("readonly {var}={value}".format(var = k, value = pipes.quote(str(v))))
''' < "${tmp_kube_env}" > "${KUBE_HOME}/kube-env")
  rm -f "${tmp_kube_env}"
}

function validate-hash {
  local -r file="$1"
  local -r expected="$2"

  actual=$(sha1sum ${file} | awk '{ print $1 }') || true
  if [[ "${actual}" != "${expected}" ]]; then
    echo "== ${file} corrupted, sha1 ${actual} doesn't match expected ${expected} =="
    return 1
  fi
}

# Retry a download until we get it. Takes a hash and a set of URLs.
#
# $1 is the sha1 of the URL. Can be "" if the sha1 is unknown.
# $2+ are the URLs to download.
function download-or-bust {
  local -r hash="$1"
  shift 1

  local -r urls=( $* )
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

function split-commas {
  echo $1 | tr "," "\n"
}

# Downloads kubernetes binaries and kube-system manifest tarball, unpacks them,
# and places them into suitable directories. Files are placed in /home/kubernetes. 
function install-kube-binary-config {
  cd "${KUBE_HOME}"
  local -r server_binary_tar_urls=( $(split-commas "${SERVER_BINARY_TAR_URL}") )
  local -r server_binary_tar="${server_binary_tar_urls[0]##*/}"
  if [[ -n "${SERVER_BINARY_TAR_HASH:-}" ]]; then
    local -r server_binary_tar_hash="${SERVER_BINARY_TAR_HASH}"
  else
    echo "Downloading binary release sha1 (not found in env)"
    download-or-bust "" "${server_binary_tar_urls[@]/.tar.gz/.tar.gz.sha1}"
    local -r server_binary_tar_hash=$(cat "${server_binary_tar}.sha1")
  fi
  echo "Downloading binary release tar"
  download-or-bust "${server_binary_tar_hash}" "${server_binary_tar_urls[@]}"
  tar xzf "${KUBE_HOME}/${server_binary_tar}" -C "${KUBE_HOME}" --overwrite
  # Copy docker_tag and image files to ${KUBE_HOME}/kube-docker-files.
  src_dir="${KUBE_HOME}/kubernetes/server/bin"
  dst_dir="${KUBE_HOME}/kube-docker-files"
  mkdir -p "${dst_dir}"
  cp "${src_dir}/"*.docker_tag "${dst_dir}"
  if [[ "${KUBERNETES_MASTER:-}" == "false" ]]; then
    cp "${src_dir}/kube-proxy.tar" "${dst_dir}"
  else
    cp "${src_dir}/kube-apiserver.tar" "${dst_dir}"
    cp "${src_dir}/kube-controller-manager.tar" "${dst_dir}"
    cp "${src_dir}/kube-scheduler.tar" "${dst_dir}"
    cp -r "${KUBE_HOME}/kubernetes/addons" "${dst_dir}"
  fi
  local -r kube_bin="${KUBE_HOME}/bin"
  # If the built-in binary version is different from the expected version, we use
  # the downloaded binary. The simplest implementation is to always use the downloaded
  # binary without checking the version. But we have another version guardian in GKE.
  # So, we compare the versions to ensure this run-time binary replacement is only
  # applied for OSS kubernetes.
  cp "${src_dir}/kubelet" "${kube_bin}"
  local -r builtin_version="$(/usr/bin/kubelet --version=true | cut -f2 -d " ")"
  local -r required_version="$(/home/kubernetes/bin/kubelet --version=true | cut -f2 -d " ")"
  if [[ "${TEST_CLUSTER:-}" == "true" ]] || \
     [[ "${builtin_version}" != "${required_version}" ]]; then
    cp "${src_dir}/kubectl" "${kube_bin}"
    chmod 755 "${kube_bin}/kubelet"
    chmod 755 "${kube_bin}/kubectl"
    mount --bind "${kube_bin}/kubelet" /usr/bin/kubelet
    mount --bind "${kube_bin}/kubectl" /usr/bin/kubectl
  else
    rm -f "${kube_bin}/kubelet"
  fi
  if [[ "${NETWORK_PROVIDER:-}" == "kubenet" ]] || \
     [[ "${NETWORK_PROVIDER:-}" == "cni" ]]; then
    #TODO(andyzheng0831): We should make the cni version number as a k8s env variable.
    local -r cni_tar="cni-8a936732094c0941e1543ef5d292a1f4fffa1ac5.tar.gz"
    download-or-bust "" "https://storage.googleapis.com/kubernetes-release/network-plugins/${cni_tar}"
    tar xzf "${KUBE_HOME}/${cni_tar}" -C "${kube_bin}" --overwrite
    mv "${kube_bin}/bin"/* "${kube_bin}"
    rmdir "${kube_bin}/bin"
    rm -f "${KUBE_HOME}/${cni_tar}"
  fi

  mv "${KUBE_HOME}/kubernetes/LICENSES" "${KUBE_HOME}"
  mv "${KUBE_HOME}/kubernetes/kubernetes-src.tar.gz" "${KUBE_HOME}"

  # Put kube-system pods manifests in ${KUBE_HOME}/kube-manifests/.
  dst_dir="${KUBE_HOME}/kube-manifests"
  mkdir -p "${dst_dir}"
  local -r manifests_tar_urls=( $(split-commas "${KUBE_MANIFESTS_TAR_URL}") )
  local -r manifests_tar="${manifests_tar_urls[0]##*/}"
  if [ -n "${KUBE_MANIFESTS_TAR_HASH:-}" ]; then
    local -r manifests_tar_hash="${KUBE_MANIFESTS_TAR_HASH}"
  else
    echo "Downloading k8s manifests sha1 (not found in env)"
    download-or-bust "" "${manifests_tar_urls[@]/.tar.gz/.tar.gz.sha1}"
    local -r manifests_tar_hash=$(cat "${manifests_tar}.sha1")
  fi
  echo "Downloading k8s manifests tar"
  download-or-bust "${manifests_tar_hash}" "${manifests_tar_urls[@]}"
  tar xzf "${KUBE_HOME}/${manifests_tar}" -C "${dst_dir}" --overwrite
  local -r kube_addon_registry="${KUBE_ADDON_REGISTRY:-gcr.io/google_containers}"
  if [[ "${kube_addon_registry}" != "gcr.io/google_containers" ]]; then
    find "${dst_dir}" -name \*.yaml -or -name \*.yaml.in | \
      xargs sed -ri "s@(image:\s.*)gcr.io/google_containers@\1${kube_addon_registry}@"
    find "${dst_dir}" -name \*.manifest -or -name \*.json | \
      xargs sed -ri "s@(image\":\s+\")gcr.io/google_containers@\1${kube_addon_registry}@"
  fi
  cp "${dst_dir}/kubernetes/gci-trusty/gci-configure-helper.sh" "${KUBE_HOME}/bin/configure-helper.sh"
  cp "${dst_dir}/kubernetes/gci-trusty/health-monitor.sh" "${KUBE_HOME}/bin/health-monitor.sh"
  chmod 544 "${KUBE_HOME}/bin/configure-helper.sh"
  chmod 544 "${KUBE_HOME}/bin/health-monitor.sh"

  # Clean up.
  rm -rf "${KUBE_HOME}/kubernetes"
  rm -f "${KUBE_HOME}/${server_binary_tar}"
  rm -f "${KUBE_HOME}/${server_binary_tar}.sha1"
  rm -f "${KUBE_HOME}/${manifests_tar}"
  rm -f "${KUBE_HOME}/${manifests_tar}.sha1"
}


######### Main Function ##########
echo "Start to install kubernetes files"
set-broken-motd
KUBE_HOME="/home/kubernetes"
download-kube-env
source "${KUBE_HOME}/kube-env"
install-kube-binary-config
echo "Done for installing kubernetes files"
