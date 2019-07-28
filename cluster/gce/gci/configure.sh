#!/usr/bin/env bash

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

### Hardcoded constants
DEFAULT_CNI_VERSION="v0.7.5"
DEFAULT_CNI_SHA1="52e9d2de8a5f927307d9397308735658ee44ab8d"
DEFAULT_NPD_VERSION="v0.6.3"
DEFAULT_NPD_SHA1="3a6ac56be6c121f1b94450bfd1a81ad28d532369"
DEFAULT_CRICTL_VERSION="v1.14.0"
DEFAULT_CRICTL_SHA1="1f93c6183d0a4e186708efe7899da7a7bce9c736"
DEFAULT_MOUNTER_TAR_SHA="8003b798cf33c7f91320cd6ee5cec4fa22244571"
###

# Use --retry-connrefused opt only if it's supported by curl.
CURL_RETRY_CONNREFUSED=""
if curl --help | grep -q -- '--retry-connrefused'; then
  CURL_RETRY_CONNREFUSED='--retry-connrefused'
fi

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
  (
    umask 077
    local -r tmp_kube_env="/tmp/kube-env.yaml"
    curl --fail --retry 5 --retry-delay 3 ${CURL_RETRY_CONNREFUSED} --silent --show-error \
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
  )
}

function download-kubelet-config {
  local -r dest="$1"
  echo "Downloading Kubelet config file, if it exists"
  # Fetch kubelet config file from GCE metadata server.
  (
    umask 077
    local -r tmp_kubelet_config="/tmp/kubelet-config.yaml"
    if curl --fail --retry 5 --retry-delay 3 ${CURL_RETRY_CONNREFUSED} --silent --show-error \
        -H "X-Google-Metadata-Request: True" \
        -o "${tmp_kubelet_config}" \
        http://metadata.google.internal/computeMetadata/v1/instance/attributes/kubelet-config; then
      # only write to the final location if curl succeeds
      mv "${tmp_kubelet_config}" "${dest}"
    elif [[ "${REQUIRE_METADATA_KUBELET_CONFIG_FILE:-false}" == "true" ]]; then
      echo "== Failed to download required Kubelet config file from metadata server =="
      exit 1
    fi
  )
}

function download-kube-master-certs {
  # Fetch kube-env from GCE metadata server.
  (
    umask 077
    local -r tmp_kube_master_certs="/tmp/kube-master-certs.yaml"
    curl --fail --retry 5 --retry-delay 3 ${CURL_RETRY_CONNREFUSED} --silent --show-error \
      -H "X-Google-Metadata-Request: True" \
      -o "${tmp_kube_master_certs}" \
      http://metadata.google.internal/computeMetadata/v1/instance/attributes/kube-master-certs
    # Convert the yaml format file into a shell-style file.
    eval $(python -c '''
import pipes,sys,yaml
for k,v in yaml.load(sys.stdin).iteritems():
  print("readonly {var}={value}".format(var = k, value = pipes.quote(str(v))))
''' < "${tmp_kube_master_certs}" > "${KUBE_HOME}/kube-master-certs")
    rm -f "${tmp_kube_master_certs}"
  )
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

# Get default service account credentials of the VM.
GCE_METADATA_INTERNAL="http://metadata.google.internal/computeMetadata/v1/instance"
function get-credentials {
  curl --fail --retry 5 --retry-delay 3 ${CURL_RETRY_CONNREFUSED} --silent --show-error "${GCE_METADATA_INTERNAL}/service-accounts/default/token" -H "Metadata-Flavor: Google" -s | python -c \
    'import sys; import json; print(json.loads(sys.stdin.read())["access_token"])'
}

function valid-storage-scope {
  curl --fail --retry 5 --retry-delay 3 ${CURL_RETRY_CONNREFUSED} --silent --show-error "${GCE_METADATA_INTERNAL}/service-accounts/default/scopes" -H "Metadata-Flavor: Google" -s | grep -q "auth/devstorage"
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
      # if the url belongs to GCS API we should use oauth2_token in the headers
      local curl_headers=""
      if [[ "$url" =~ ^https://storage.googleapis.com.* ]] && valid-storage-scope ; then
        curl_headers="Authorization: Bearer $(get-credentials)"
      fi
      if ! curl ${curl_headers:+-H "${curl_headers}"} -f --ipv4 -Lo "${file}" --connect-timeout 20 --max-time 300 --retry 6 --retry-delay 10 ${CURL_RETRY_CONNREFUSED} "${url}"; then
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

function is-preloaded {
  local -r key=$1
  local -r value=$2
  grep -qs "${key},${value}" "${KUBE_HOME}/preload_info"
}

function split-commas {
  echo $1 | tr "," "\n"
}

function remount-flexvolume-directory {
  local -r flexvolume_plugin_dir=$1
  mkdir -p $flexvolume_plugin_dir
  mount --bind $flexvolume_plugin_dir $flexvolume_plugin_dir
  mount -o remount,exec $flexvolume_plugin_dir
}

function install-gci-mounter-tools {
  CONTAINERIZED_MOUNTER_HOME="${KUBE_HOME}/containerized_mounter"
  local -r mounter_tar_sha="${DEFAULT_MOUNTER_TAR_SHA}"
  if is-preloaded "mounter" "${mounter_tar_sha}"; then
    echo "mounter is preloaded."
    return
  fi

  echo "Downloading gci mounter tools."
  mkdir -p "${CONTAINERIZED_MOUNTER_HOME}"
  chmod a+x "${CONTAINERIZED_MOUNTER_HOME}"
  mkdir -p "${CONTAINERIZED_MOUNTER_HOME}/rootfs"
  download-or-bust "${mounter_tar_sha}" "https://storage.googleapis.com/kubernetes-release/gci-mounter/mounter.tar"
  cp "${KUBE_HOME}/kubernetes/server/bin/mounter" "${CONTAINERIZED_MOUNTER_HOME}/mounter"
  chmod a+x "${CONTAINERIZED_MOUNTER_HOME}/mounter"
  mv "${KUBE_HOME}/mounter.tar" /tmp/mounter.tar
  tar xf /tmp/mounter.tar -C "${CONTAINERIZED_MOUNTER_HOME}/rootfs"
  rm /tmp/mounter.tar
  mkdir -p "${CONTAINERIZED_MOUNTER_HOME}/rootfs/var/lib/kubelet"
}

# Install node problem detector binary.
function install-node-problem-detector {
  if [[ -n "${NODE_PROBLEM_DETECTOR_VERSION:-}" ]]; then
      local -r npd_version="${NODE_PROBLEM_DETECTOR_VERSION}"
      local -r npd_sha1="${NODE_PROBLEM_DETECTOR_TAR_HASH}"
  else
      local -r npd_version="${DEFAULT_NPD_VERSION}"
      local -r npd_sha1="${DEFAULT_NPD_SHA1}"
  fi
  local -r npd_tar="node-problem-detector-${npd_version}.tar.gz"

  if is-preloaded "${npd_tar}" "${npd_sha1}"; then
    echo "${npd_tar} is preloaded."
    return
  fi

  echo "Downloading ${npd_tar}."
  local -r npd_release_path="${NODE_PROBLEM_DETECTOR_RELEASE_PATH:-https://storage.googleapis.com/kubernetes-release}"
  download-or-bust "${npd_sha1}" "${npd_release_path}/node-problem-detector/${npd_tar}"
  local -r npd_dir="${KUBE_HOME}/node-problem-detector"
  mkdir -p "${npd_dir}"
  tar xzf "${KUBE_HOME}/${npd_tar}" -C "${npd_dir}" --overwrite
  mv "${npd_dir}/bin"/* "${KUBE_BIN}"
  chmod a+x "${KUBE_BIN}/node-problem-detector"
  rmdir "${npd_dir}/bin"
  rm -f "${KUBE_HOME}/${npd_tar}"
}

function install-cni-binaries {
  if [[ -n "${CNI_VERSION:-}" ]]; then
      local -r cni_tar="cni-plugins-amd64-${CNI_VERSION}.tgz"
      local -r cni_sha1="${CNI_SHA1}"
  else
      local -r cni_tar="cni-plugins-amd64-${DEFAULT_CNI_VERSION}.tgz"
      local -r cni_sha1="${DEFAULT_CNI_SHA1}"
  fi
  if is-preloaded "${cni_tar}" "${cni_sha1}"; then
    echo "${cni_tar} is preloaded."
    return
  fi

  echo "Downloading cni binaries"
  download-or-bust "${cni_sha1}" "${CNI_STORAGE_PATH}/${cni_tar}"
  local -r cni_dir="${KUBE_HOME}/cni"
  mkdir -p "${cni_dir}/bin"
  tar xzf "${KUBE_HOME}/${cni_tar}" -C "${cni_dir}/bin" --overwrite
  mv "${cni_dir}/bin"/* "${KUBE_BIN}"
  rmdir "${cni_dir}/bin"
  rm -f "${KUBE_HOME}/${cni_tar}"
}

# Install crictl binary.
function install-crictl {
  if [[ -n "${CRICTL_VERSION:-}" ]]; then
    local -r crictl_version="${CRICTL_VERSION}"
    local -r crictl_sha1="${CRICTL_TAR_HASH}"
  else
    local -r crictl_version="${DEFAULT_CRICTL_VERSION}"
    local -r crictl_sha1="${DEFAULT_CRICTL_SHA1}"
  fi
  local -r crictl="crictl-${crictl_version}-linux-amd64"

  # Create crictl config file.
  cat > /etc/crictl.yaml <<EOF
runtime-endpoint: ${CONTAINER_RUNTIME_ENDPOINT:-unix:///var/run/dockershim.sock}
EOF

  if is-preloaded "${crictl}" "${crictl_sha1}"; then
    echo "crictl is preloaded"
    return
  fi

  echo "Downloading crictl"
  local -r crictl_path="https://storage.googleapis.com/kubernetes-release/crictl"
  download-or-bust "${crictl_sha1}" "${crictl_path}/${crictl}"
  mv "${KUBE_HOME}/${crictl}" "${KUBE_BIN}/crictl"
  chmod a+x "${KUBE_BIN}/crictl"
}

function install-exec-auth-plugin {
  if [[ ! "${EXEC_AUTH_PLUGIN_URL:-}" ]]; then
      return
  fi
  local -r plugin_url="${EXEC_AUTH_PLUGIN_URL}"
  local -r plugin_sha1="${EXEC_AUTH_PLUGIN_SHA1}"

  echo "Downloading gke-exec-auth-plugin binary"
  download-or-bust "${plugin_sha1}" "${plugin_url}"
  mv "${KUBE_HOME}/gke-exec-auth-plugin" "${KUBE_BIN}/gke-exec-auth-plugin"
  chmod a+x "${KUBE_BIN}/gke-exec-auth-plugin"

  if [[ ! "${EXEC_AUTH_PLUGIN_LICENSE_URL:-}" ]]; then
      return
  fi
  local -r license_url="${EXEC_AUTH_PLUGIN_LICENSE_URL}"
  echo "Downloading gke-exec-auth-plugin license"
  download-or-bust "" "${license_url}"
  mv "${KUBE_HOME}/LICENSE" "${KUBE_BIN}/gke-exec-auth-plugin-license"
}

function install-kube-manifests {
  # Put kube-system pods manifests in ${KUBE_HOME}/kube-manifests/.
  local dst_dir="${KUBE_HOME}/kube-manifests"
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

  if is-preloaded "${manifests_tar}" "${manifests_tar_hash}"; then
    echo "${manifests_tar} is preloaded."
    return
  fi

  echo "Downloading k8s manifests tar"
  download-or-bust "${manifests_tar_hash}" "${manifests_tar_urls[@]}"
  tar xzf "${KUBE_HOME}/${manifests_tar}" -C "${dst_dir}" --overwrite
  local -r kube_addon_registry="${KUBE_ADDON_REGISTRY:-k8s.gcr.io}"
  if [[ "${kube_addon_registry}" != "k8s.gcr.io" ]]; then
    find "${dst_dir}" -name \*.yaml -or -name \*.yaml.in | \
      xargs sed -ri "s@(image:\s.*)k8s.gcr.io@\1${kube_addon_registry}@"
    find "${dst_dir}" -name \*.manifest -or -name \*.json | \
      xargs sed -ri "s@(image\":\s+\")k8s.gcr.io@\1${kube_addon_registry}@"
  fi
  cp "${dst_dir}/kubernetes/gci-trusty/gci-configure-helper.sh" "${KUBE_BIN}/configure-helper.sh"
  if [[ -e "${dst_dir}/kubernetes/gci-trusty/gke-internal-configure-helper.sh" ]]; then
    cp "${dst_dir}/kubernetes/gci-trusty/gke-internal-configure-helper.sh" "${KUBE_BIN}/"
  fi

  cp "${dst_dir}/kubernetes/gci-trusty/health-monitor.sh" "${KUBE_BIN}/health-monitor.sh"

  rm -f "${KUBE_HOME}/${manifests_tar}"
  rm -f "${KUBE_HOME}/${manifests_tar}.sha1"
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
  until timeout 30 ${LOAD_IMAGE_COMMAND:-docker load -i} "${img}"; do
    if [[ "${attempt_num}" == "${max_attempts}" ]]; then
      echo "Fail to load docker image file ${img} after ${max_attempts} retries. Exit!!"
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

  if is-preloaded "${server_binary_tar}" "${server_binary_tar_hash}"; then
    echo "${server_binary_tar} is preloaded."
  else
    echo "Downloading binary release tar"
    download-or-bust "${server_binary_tar_hash}" "${server_binary_tar_urls[@]}"
    tar xzf "${KUBE_HOME}/${server_binary_tar}" -C "${KUBE_HOME}" --overwrite
    # Copy docker_tag and image files to ${KUBE_HOME}/kube-docker-files.
    local -r src_dir="${KUBE_HOME}/kubernetes/server/bin"
    local dst_dir="${KUBE_HOME}/kube-docker-files"
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
    load-docker-images
    mv "${src_dir}/kubelet" "${KUBE_BIN}"
    mv "${src_dir}/kubectl" "${KUBE_BIN}"

    mv "${KUBE_HOME}/kubernetes/LICENSES" "${KUBE_HOME}"
    mv "${KUBE_HOME}/kubernetes/kubernetes-src.tar.gz" "${KUBE_HOME}"
  fi

  if [[ "${KUBERNETES_MASTER:-}" == "false" ]] && \
     [[ "${ENABLE_NODE_PROBLEM_DETECTOR:-}" == "standalone" ]]; then
    install-node-problem-detector
  fi

  if [[ "${NETWORK_PROVIDER:-}" == "kubenet" ]] || \
     [[ "${NETWORK_PROVIDER:-}" == "cni" ]]; then
    install-cni-binaries
  fi

  # Put kube-system pods manifests in ${KUBE_HOME}/kube-manifests/.
  install-kube-manifests
  chmod -R 755 "${KUBE_BIN}"

  # Install gci mounter related artifacts to allow mounting storage volumes in GCI
  install-gci-mounter-tools

  # Remount the Flexvolume directory with the "exec" option, if needed.
  if [[ "${REMOUNT_VOLUME_PLUGIN_DIR:-}" == "true" && -n "${VOLUME_PLUGIN_DIR:-}" ]]; then
    remount-flexvolume-directory "${VOLUME_PLUGIN_DIR}"
  fi

  # Install crictl on each node.
  install-crictl

  # TODO(awly): include the binary and license in the OS image.
  install-exec-auth-plugin

  # Clean up.
  rm -rf "${KUBE_HOME}/kubernetes"
  rm -f "${KUBE_HOME}/${server_binary_tar}"
  rm -f "${KUBE_HOME}/${server_binary_tar}.sha1"
}

######### Main Function ##########
echo "Start to install kubernetes files"
# if install fails, message-of-the-day (motd) will warn at login shell
set-broken-motd

KUBE_HOME="/home/kubernetes"
KUBE_BIN="${KUBE_HOME}/bin"

# download and source kube-env
download-kube-env
source "${KUBE_HOME}/kube-env"

download-kubelet-config "${KUBE_HOME}/kubelet-config.yaml"

# master certs
if [[ "${KUBERNETES_MASTER:-}" == "true" ]]; then
  download-kube-master-certs
fi

# binaries and kube-system manifests
install-kube-binary-config

echo "Done for installing kubernetes files"
