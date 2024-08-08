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
DEFAULT_CNI_VERSION='v1.5.0'
DEFAULT_CNI_HASH='7e3d34d2de37a8fe72278ea796fcbe763e9e7f241ed1a6d3c5a9f613fe5895bf3bbea05e0b651d9e4d0e1c427a7b42f0ff5c07d29abfc2aee9967136347451bd'
DEFAULT_NPD_VERSION='v0.8.19'
DEFAULT_NPD_HASH_AMD64='41e7816614a1e30c94cbe37a1d09c2ef6c3f2ad336b40d95378528deac9a8b811356e46bae268dab006132f2bdce22668a640517f000853109d1e3be77aef25e'
DEFAULT_NPD_HASH_ARM64='678352d0544c166e86d3435c4f49b1ea5ce19c2d6f083f51f7688599377928e5b1425d1733006981d17e94ddede67f9fc1d2035577a87687701044950f697390'
DEFAULT_CRICTL_VERSION='v1.31.1'
DEFAULT_CRICTL_AMD64_SHA512='831ee7b3589197dbee399973793e0750e9870cd963e0d6c57eca9231fbc366c2e683855cdcabede33acdb56c15161cc9d40d5a01ec2de8cfee21ba8aa8adba54'
DEFAULT_CRICTL_ARM64_SHA512='4d12cf190c03d03d86a1a10b93abbbcb4857d013b62a601b17d767d2397d3e17f7c93d3d32b54cc1ac80262c0837afa5f34c88a58bf52d93e5cfc330dd83218c'
DEFAULT_MOUNTER_TAR_SHA='7956fd42523de6b3107ddc3ce0e75233d2fcb78436ff07a1389b6eaac91fb2b1b72a08f7a219eaf96ba1ca4da8d45271002e0d60e0644e796c665f99bb356516'
AUTH_PROVIDER_GCP_HASH_LINUX_AMD64="${AUTH_PROVIDER_GCP_HASH_LINUX_AMD64:-156058e5b3994cba91c23831774033e0d505d6d8b80f43541ef6af91b320fd9dfaabe42ec8a8887b51d87104c2b57e1eb895649d681575ffc80dd9aee8e563db}"
AUTH_PROVIDER_GCP_HASH_LINUX_ARM64="${AUTH_PROVIDER_GCP_HASH_LINUX_ARM64:-1aa3b0bea10a9755231989ffc150cbfa770f1d96932db7535473f7bfeb1108bafdae80202ae738d59495982512e716ff7366d5f414d0e76dd50519f98611f9ab}"
###

# Standard curl flags.
CURL_FLAGS='--fail --silent --show-error --retry 5 --retry-delay 3 --connect-timeout 10 --retry-connrefused'

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

# A function that fetches a GCE metadata value and echoes it out.
# Args:
#   $1 : URL path after /computeMetadata/v1/ (without heading slash).
#   $2 : An optional default value to echo out if the fetch fails.
#
# NOTE: this function is duplicated in configure-helper.sh, any changes here
# should be duplicated there as well.
function get-metadata-value {
  local default="${2:-}"

  local status
  # shellcheck disable=SC2086
  curl ${CURL_FLAGS} \
    -H 'Metadata-Flavor: Google' \
    "http://metadata/computeMetadata/v1/${1}" \
  || status="$?"
  status="${status:-0}"

  if [[ "${status}" -eq 0 || -z "${default}" ]]; then
    return "${status}"
  else
    echo "${default}"
  fi
}

function download-kube-env {
  # Fetch kube-env from GCE metadata server.
  (
    umask 077
    local -r tmp_kube_env="/tmp/kube-env.yaml"
    # shellcheck disable=SC2086
    retry-forever 10 curl ${CURL_FLAGS} \
      -H "X-Google-Metadata-Request: True" \
      -o "${tmp_kube_env}" \
      http://metadata.google.internal/computeMetadata/v1/instance/attributes/kube-env
    # Convert the yaml format file into a shell-style file.
    eval "$(python3 -c '''
import pipes,sys,yaml
items = yaml.load(sys.stdin, Loader=yaml.BaseLoader).items()
for k, v in items:
    print("readonly {var}={value}".format(var=k, value=pipes.quote(str(v))))
''' < "${tmp_kube_env}" > "${KUBE_HOME}/kube-env")"
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
    # shellcheck disable=SC2086
    retry-forever 10 curl ${CURL_FLAGS} \
      -H "X-Google-Metadata-Request: True" \
      -o "${tmp_kubelet_config}" \
      http://metadata.google.internal/computeMetadata/v1/instance/attributes/kubelet-config
    # only write to the final location if curl succeeds
    mv "${tmp_kubelet_config}" "${dest}"
  )
}

function download-kube-master-certs {
  # Fetch kube-env from GCE metadata server.
  (
    umask 077
    local -r tmp_kube_master_certs="/tmp/kube-master-certs.yaml"
    # shellcheck disable=SC2086
    retry-forever 10 curl ${CURL_FLAGS} \
      -H "X-Google-Metadata-Request: True" \
      -o "${tmp_kube_master_certs}" \
      http://metadata.google.internal/computeMetadata/v1/instance/attributes/kube-master-certs
    # Convert the yaml format file into a shell-style file.
    eval "$(python3 -c '''
import pipes,sys,yaml
items = yaml.load(sys.stdin, Loader=yaml.BaseLoader).items()
for k, v in items:
    print("readonly {var}={value}".format(var=k, value=pipes.quote(str(v))))
''' < "${tmp_kube_master_certs}" > "${KUBE_HOME}/kube-master-certs")"
    rm -f "${tmp_kube_master_certs}"
  )
}

function validate-hash {
  local -r file="$1"
  local -r expected="$2"

  actual_sha1=$(sha1sum "${file}" | awk '{ print $1 }') || true
  actual_sha512=$(sha512sum "${file}" | awk '{ print $1 }') || true
  if [[ "${actual_sha1}" != "${expected}" ]] && [[ "${actual_sha512}" != "${expected}" ]]; then
    echo "== ${file} corrupted, sha1 ${actual_sha1}/sha512 ${actual_sha512} doesn't match expected ${expected} =="
    return 1
  fi
}

# Get default service account credentials of the VM.
GCE_METADATA_INTERNAL="http://metadata.google.internal/computeMetadata/v1/instance"
function get-credentials {
  # shellcheck disable=SC2086
  curl ${CURL_FLAGS} \
    -H "Metadata-Flavor: Google" \
    "${GCE_METADATA_INTERNAL}/service-accounts/default/token" \
  | python3 -c 'import sys; import json; print(json.loads(sys.stdin.read())["access_token"])'
}

function valid-storage-scope {
  # shellcheck disable=SC2086
  curl ${CURL_FLAGS} \
    -H "Metadata-Flavor: Google" \
    "${GCE_METADATA_INTERNAL}/service-accounts/default/scopes" \
  | grep -E "auth/devstorage|auth/cloud-platform"
}

# Retry a download until we get it. Takes a hash and a set of URLs.
#
# $1 is the sha512/sha1 hash of the URL. Can be "" if the sha512/sha1 hash is unknown.
# $2+ are the URLs to download.
function download-or-bust {
  local -r hash="$1"
  shift 1

  while true; do
    for url in "$@"; do
      local file="${url##*/}"
      rm -f "${file}"
      # if the url belongs to GCS API we should use oauth2_token in the headers if the VM service account has storage scopes
      local curl_headers=""

      if [[ "$url" =~ ^https://storage.googleapis.com.* ]] ; then
        local canUseCredentials=0

        echo "Getting the scope of service account configured for VM."
        if ! valid-storage-scope ; then
          canUseCredentials=1
          # this behavior is preserved for backward compatibility. We want to fail fast if SA is not available
          # and try to download without SA if scope does not exist on SA
          echo "No service account or service account without storage scope. Attempt to download without service account token."
        fi

        if [[ "${canUseCredentials}" == "0" ]] ; then
          echo "Getting the service account access token configured for VM."
          local access_token="";
          if access_token=$(get-credentials); then
            echo "Service account access token is received. Downloading ${url} using this token."
          else
            echo "Cannot get a service account token. Exiting."
            exit 1
          fi

          curl_headers=${access_token:+Authorization: Bearer "${access_token}"}
        fi
      fi
      if ! curl ${curl_headers:+-H "${curl_headers}"} -f --ipv4 -Lo "${file}" --connect-timeout 20 --max-time 300 --retry 6 --retry-delay 10 --retry-connrefused "${url}"; then
        echo "== Failed to download ${url}. Retrying. =="
      elif [[ -n "${hash}" ]] && ! validate-hash "${file}" "${hash}"; then
        echo "== Hash validation of ${url} failed. Retrying. =="
      else
        if [[ -n "${hash}" ]]; then
          echo "== Downloaded ${url} (HASH = ${hash}) =="
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
  echo -e "${1//,/'\n'}"
}

function remount-flexvolume-directory {
  local -r flexvolume_plugin_dir=$1
  mkdir -p "$flexvolume_plugin_dir"
  mount --bind "$flexvolume_plugin_dir" "$flexvolume_plugin_dir"
  mount -o remount,exec "$flexvolume_plugin_dir"
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
      local -r npd_hash="${NODE_PROBLEM_DETECTOR_TAR_HASH}"
  else
      local -r npd_version="${DEFAULT_NPD_VERSION}"
      case "${HOST_PLATFORM}/${HOST_ARCH}" in
        linux/amd64)
          local -r npd_hash="${DEFAULT_NPD_HASH_AMD64}"
          ;;
        linux/arm64)
          local -r npd_hash="${DEFAULT_NPD_HASH_ARM64}"
          ;;
        # no other architectures are supported currently.
        # Assumption is that this script only runs on linux,
        # see cluster/gce/windows/k8s-node-setup.psm1 for windows
        # https://github.com/kubernetes/node-problem-detector/releases/
        *)
          echo "Unrecognized version and platform/arch combination:"
          echo "$DEFAULT_NPD_VERSION $HOST_PLATFORM/$HOST_ARCH"
          echo "Set NODE_PROBLEM_DETECTOR_VERSION and NODE_PROBLEM_DETECTOR_TAR_HASH to overwrite"
          exit 1
          ;;
      esac
  fi
  local -r npd_tar="node-problem-detector-${npd_version}-${HOST_PLATFORM}_${HOST_ARCH}.tar.gz"

  if is-preloaded "${npd_tar}" "${npd_hash}"; then
    echo "${npd_tar} is preloaded."
    return
  fi

  if [[ -n "${NODE_PROBLEM_DETECTOR_RELEASE_PATH:-}" ]]; then
    echo "Downloading ${npd_tar} from ${NODE_PROBLEM_DETECTOR_RELEASE_PATH}."
    local -r download_path="${NODE_PROBLEM_DETECTOR_RELEASE_PATH}/node-problem-detector/${npd_tar}"
  else
    echo "Downloading ${npd_tar} from github."
    local -r download_path="https://github.com/kubernetes/node-problem-detector/releases/download/${npd_version}/${npd_tar}"
  fi
  download-or-bust "${npd_hash}" "${download_path}"
  local -r npd_dir="${KUBE_HOME}/node-problem-detector"
  mkdir -p "${npd_dir}"
  tar xzf "${KUBE_HOME}/${npd_tar}" -C "${npd_dir}" --overwrite
  mv "${npd_dir}/bin"/* "${KUBE_BIN}"
  chmod a+x "${KUBE_BIN}/node-problem-detector"
  rmdir "${npd_dir}/bin"
  rm -f "${KUBE_HOME}/${npd_tar}"
}

function install-cni-binaries {
  local -r cni_version=${CNI_VERSION:-$DEFAULT_CNI_VERSION}
  if [[ -n "${CNI_VERSION:-}" ]]; then
      local -r cni_hash="${CNI_HASH:-}"
  else
      local -r cni_hash="${DEFAULT_CNI_HASH}"
  fi

  local -r cni_tar="${CNI_TAR_PREFIX}${cni_version}.tgz"
  local -r cni_url="${CNI_STORAGE_URL_BASE}/${cni_version}/${cni_tar}"

  if is-preloaded "${cni_tar}" "${cni_hash}"; then
    echo "${cni_tar} is preloaded."
    return
  fi

  echo "Downloading cni binaries"
  download-or-bust "${cni_hash}" "${cni_url}"
  local -r cni_dir="${KUBE_HOME}/cni"
  mkdir -p "${cni_dir}/bin"
  tar xzf "${KUBE_HOME}/${cni_tar}" -C "${cni_dir}/bin" --overwrite
  mv "${cni_dir}/bin"/* "${KUBE_BIN}"
  rmdir "${cni_dir}/bin"
  rm -f "${KUBE_HOME}/${cni_tar}"
}

# Install crictl binary.
# Assumptions: HOST_PLATFORM and HOST_ARCH are specified by calling detect_host_info.
function install-crictl {
  if [[ -n "${CRICTL_VERSION:-}" ]]; then
    local -r crictl_version="${CRICTL_VERSION}"
    local -r crictl_hash="${CRICTL_TAR_HASH}"
  else
    local -r crictl_version="${DEFAULT_CRICTL_VERSION}"
    case "${HOST_PLATFORM}/${HOST_ARCH}" in
      linux/amd64)
        local -r crictl_hash="${DEFAULT_CRICTL_AMD64_SHA512}"
        ;;
      linux/arm64)
        local -r crictl_hash="${DEFAULT_CRICTL_ARM64_SHA512}"
        ;;
      *)
        echo "Unrecognized version and platform/arch combination:"
        echo "$DEFAULT_CRICTL_VERSION $HOST_PLATFORM/$HOST_ARCH"
        echo "Set CRICTL_VERSION and CRICTL_TAR_HASH to overwrite"
        exit 1
    esac
  fi
  local -r crictl="crictl-${crictl_version}-${HOST_PLATFORM}-${HOST_ARCH}.tar.gz"

  # Create crictl config file.
  cat > /etc/crictl.yaml <<EOF
runtime-endpoint: ${CONTAINER_RUNTIME_ENDPOINT:-unix:///run/containerd/containerd.sock}
EOF

  if is-preloaded "${crictl}" "${crictl_hash}"; then
    echo "crictl is preloaded"
    return
  fi

  echo "Downloading crictl"
  local -r crictl_path="https://storage.googleapis.com/k8s-artifacts-cri-tools/release/${crictl_version}"
  download-or-bust "${crictl_hash}" "${crictl_path}/${crictl}"
  tar xf "${crictl}"
  mv crictl "${KUBE_BIN}/crictl"
  rm -f "${crictl}"
}

function install-kube-manifests {
  # Put kube-system pods manifests in ${KUBE_HOME}/kube-manifests/.
  local dst_dir="${KUBE_HOME}/kube-manifests"
  mkdir -p "${dst_dir}"
  local manifests_tar_urls
  while IFS= read -r url; do
    manifests_tar_urls+=("$url")
  done < <(split-commas "${KUBE_MANIFESTS_TAR_URL}")
  local -r manifests_tar="${manifests_tar_urls[0]##*/}"
  if [ -n "${KUBE_MANIFESTS_TAR_HASH:-}" ]; then
    local -r manifests_tar_hash="${KUBE_MANIFESTS_TAR_HASH}"
  else
    echo "Downloading k8s manifests hash (not found in env)"
    download-or-bust "" "${manifests_tar_urls[@]/.tar.gz/.tar.gz.sha512}"
    local -r manifests_tar_hash=$(cat "${manifests_tar}.sha512")
  fi

  if is-preloaded "${manifests_tar}" "${manifests_tar_hash}"; then
    echo "${manifests_tar} is preloaded."
    return
  fi

  echo "Downloading k8s manifests tar"
  download-or-bust "${manifests_tar_hash}" "${manifests_tar_urls[@]}"
  tar xzf "${KUBE_HOME}/${manifests_tar}" -C "${dst_dir}" --overwrite
  local -r kube_addon_registry="${KUBE_ADDON_REGISTRY:-registry.k8s.io}"
  if [[ "${kube_addon_registry}" != "registry.k8s.io" ]]; then
    find "${dst_dir}" \( -name '*.yaml' -or -name '*.yaml.in' \) -print0 | \
      xargs -0 sed -ri "s@(image:\s.*)registry.k8s.io@\1${kube_addon_registry}@"
    find "${dst_dir}" \( -name '*.manifest' -or -name '*.json' \) -print0 | \
      xargs -0 sed -ri "s@(image\":\s+\")registry.k8s.io@\1${kube_addon_registry}@"
  fi
  cp "${dst_dir}/kubernetes/gci-trusty/gci-configure-helper.sh" "${KUBE_BIN}/configure-helper.sh"
  cp "${dst_dir}/kubernetes/gci-trusty/configure-kubeapiserver.sh" "${KUBE_BIN}/configure-kubeapiserver.sh"
  if [[ -e "${dst_dir}/kubernetes/gci-trusty/gke-internal-configure-helper.sh" ]]; then
    cp "${dst_dir}/kubernetes/gci-trusty/gke-internal-configure-helper.sh" "${KUBE_BIN}/"
  fi

  cp "${dst_dir}/kubernetes/gci-trusty/health-monitor.sh" "${KUBE_BIN}/health-monitor.sh"

  rm -f "${KUBE_HOME}/${manifests_tar}"
  rm -f "${KUBE_HOME}/${manifests_tar}.sha512"
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

  if [[ "${CONTAINER_RUNTIME_NAME:-}" == "containerd" || "${CONTAINERD_TEST:-}"  == "containerd" ]]; then
    load_image_command=${LOAD_IMAGE_COMMAND:-ctr -n=k8s.io images import}
    tag_image_command=${TAG_IMAGE_COMMAND:-ctr -n=k8s.io images tag}
  else
    load_image_command="${LOAD_IMAGE_COMMAND:-}"
    tag_image_command="${TAG_IMAGE_COMMAND:-}"
  fi

  # Deliberately word split load_image_command
  # shellcheck disable=SC2086
  until timeout 30 ${load_image_command} "${img}"; do
    if [[ "${attempt_num}" == "${max_attempts}" ]]; then
      echo "Fail to load docker image file ${img} using ${load_image_command} after ${max_attempts} retries. Exit!!"
      exit 1
    else
      attempt_num=$((attempt_num+1))
      sleep 5
    fi
  done

  if [[ -n "${KUBE_ADDON_REGISTRY:-}" ]]; then
    # remove the prefix and suffix from the path to get the container name
    container=${img##*/}
    container=${container%.tar}
    # find the right one for which we will need an additional tag
    container=$(ctr -n k8s.io images ls | grep "registry.k8s.io/${container}" | awk '{print $1}' | cut -f 2 -d '/')
    ${tag_image_command} "registry.k8s.io/${container}" "${KUBE_ADDON_REGISTRY}/${container}"
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

# If we are on ubuntu we can try to install containerd
function install-containerd-ubuntu {
  # bailout if we are not on ubuntu
  if [[ -z "$(command -v lsb_release)" || $(lsb_release -si) != "Ubuntu" ]]; then
    echo "Unable to automatically install containerd in non-ubuntu image. Bailing out..."
    exit 2
  fi

  # Install dependencies, some of these are already installed in the image but
  # that's fine since they won't re-install and we can reuse the code below
  # for another image someday.
  apt-get update
  apt-get install -y --no-install-recommends \
    apt-transport-https \
    ca-certificates \
    socat \
    curl \
    gnupg2 \
    software-properties-common \
    lsb-release

  release=$(lsb_release -cs)

  # Add the Docker apt-repository (as we install containerd from there)
  # shellcheck disable=SC2086
  curl ${CURL_FLAGS} \
    --location \
    "https://download.docker.com/${HOST_PLATFORM}/$(. /etc/os-release; echo "$ID")/gpg" \
  | apt-key add -
  add-apt-repository \
    "deb [arch=${HOST_ARCH}] https://download.docker.com/${HOST_PLATFORM}/$(. /etc/os-release; echo "$ID") \
    $release stable"

  # Install containerd from Docker repo
  apt-get update && \
    apt-get install -y --no-install-recommends containerd
  rm -rf /var/lib/apt/lists/*

  # Override to latest versions of containerd and runc
  systemctl stop containerd
  if [[ -n "${UBUNTU_INSTALL_CONTAINERD_VERSION:-}" ]]; then
    # containerd versions have slightly different url(s), so try both
    # shellcheck disable=SC2086
    ( curl ${CURL_FLAGS} \
        --location \
        "https://github.com/containerd/containerd/releases/download/${UBUNTU_INSTALL_CONTAINERD_VERSION}/containerd-${UBUNTU_INSTALL_CONTAINERD_VERSION:1}-${HOST_PLATFORM}-${HOST_ARCH}.tar.gz" \
      || curl ${CURL_FLAGS} \
        --location \
        "https://github.com/containerd/containerd/releases/download/${UBUNTU_INSTALL_CONTAINERD_VERSION}/containerd-${UBUNTU_INSTALL_CONTAINERD_VERSION:1}.${HOST_PLATFORM}-${HOST_ARCH}.tar.gz" ) \
    | tar --overwrite -xzv -C /usr/
  fi
  if [[ -n "${UBUNTU_INSTALL_RUNC_VERSION:-}" ]]; then
    # shellcheck disable=SC2086
    curl ${CURL_FLAGS} \
      --location \
      "https://github.com/opencontainers/runc/releases/download/${UBUNTU_INSTALL_RUNC_VERSION}/runc.${HOST_ARCH}" --output /usr/sbin/runc \
    && chmod 755 /usr/sbin/runc
  fi
  sudo systemctl start containerd
}

# If we are on cos we can try to install containerd
function install-containerd-cos {
  # bailout if we are not on COS
  if [ -e /etc/os-release ] && ! grep -q "ID=cos" /etc/os-release; then
    echo "Unable to automatically install containerd in non-cos image. Bailing out..."
    exit 2
  fi

  # Override to latest versions of containerd and runc
  systemctl stop containerd
  mkdir -p /home/containerd/
  mount --bind /home/containerd /home/containerd
  mount -o remount,exec /home/containerd
  if [[ -n "${COS_INSTALL_CONTAINERD_VERSION:-}" ]]; then
    # containerd versions have slightly different url(s), so try both
    # shellcheck disable=SC2086
    ( curl ${CURL_FLAGS} \
        --location \
        "https://github.com/containerd/containerd/releases/download/${COS_INSTALL_CONTAINERD_VERSION}/containerd-${COS_INSTALL_CONTAINERD_VERSION:1}-${HOST_PLATFORM}-${HOST_ARCH}.tar.gz" \
      || curl ${CURL_FLAGS} \
        --location \
        "https://github.com/containerd/containerd/releases/download/${COS_INSTALL_CONTAINERD_VERSION}/containerd-${COS_INSTALL_CONTAINERD_VERSION:1}.${HOST_PLATFORM}-${HOST_ARCH}.tar.gz" ) \
    | tar --overwrite -xzv -C /home/containerd/
    cp /usr/lib/systemd/system/containerd.service /etc/systemd/system/containerd.service
    # fix the path of the new containerd binary
    sed -i 's|ExecStart=.*|ExecStart=/home/containerd/bin/containerd|' /etc/systemd/system/containerd.service
  fi
  if [[ -n "${COS_INSTALL_RUNC_VERSION:-}" ]]; then
    # shellcheck disable=SC2086
    curl ${CURL_FLAGS} \
      --location \
      "https://github.com/opencontainers/runc/releases/download/${COS_INSTALL_RUNC_VERSION}/runc.${HOST_ARCH}" --output /home/containerd/bin/runc \
    && chmod 755 /home/containerd/bin/runc
    # ensure runc gets picked up from the correct location
    sed -i "/\[Service\]/a Environment=PATH=/home/containerd/bin:$PATH" /etc/systemd/system/containerd.service
  fi
  systemctl daemon-reload
  sudo systemctl start containerd
}

function install-auth-provider-gcp {
  local -r filename="auth-provider-gcp"
  local -r auth_provider_storage_full_path="${AUTH_PROVIDER_GCP_STORAGE_PATH}/${AUTH_PROVIDER_GCP_VERSION}/${HOST_PLATFORM}_${HOST_ARCH}/${filename}"
  echo "Downloading auth-provider-gcp ${auth_provider_storage_full_path}" .

  case "${HOST_ARCH}" in
    amd64)
      local -r auth_provider_gcp_hash="${AUTH_PROVIDER_GCP_HASH_LINUX_AMD64}"
      ;;
    arm64)
      local -r auth_provider_gcp_hash="${AUTH_PROVIDER_GCP_HASH_LINUX_ARM64}"
      ;;
    *)
      echo "Unrecognized version and platform/arch combination: ${HOST_PLATFORM}/${HOST_ARCH}"
      exit 1
  esac

  download-or-bust "${auth_provider_gcp_hash}" "${auth_provider_storage_full_path}"

  mv "${KUBE_HOME}/${filename}" "${AUTH_PROVIDER_GCP_LINUX_BIN_DIR}"
  chmod a+x "${AUTH_PROVIDER_GCP_LINUX_BIN_DIR}/${filename}"

  cat >> "${AUTH_PROVIDER_GCP_LINUX_CONF_FILE}" << EOF
kind: CredentialProviderConfig
apiVersion: kubelet.config.k8s.io/v1
providers:
  - name: auth-provider-gcp
    apiVersion: credentialprovider.kubelet.k8s.io/v1
    matchImages:
    - "container.cloud.google.com"
    - "gcr.io"
    - "*.gcr.io"
    - "*.pkg.dev"
    args:
    - get-credentials
    - --v=3
    defaultCacheDuration: 1m
EOF
}

function ensure-containerd-runtime {
  # Install containerd/runc if requested
  if [[ -n "${UBUNTU_INSTALL_CONTAINERD_VERSION:-}" || -n "${UBUNTU_INSTALL_RUNC_VERSION:-}" ]]; then
    log-wrap "InstallContainerdUbuntu" install-containerd-ubuntu
  fi
  if [[ -n "${COS_INSTALL_CONTAINERD_VERSION:-}" || -n "${COS_INSTALL_RUNC_VERSION:-}" ]]; then
    log-wrap "InstallContainerdCOS" install-containerd-cos
  fi

  # Fall back to installing distro specific containerd, if not found
  if ! command -v containerd >/dev/null 2>&1; then
    local linuxrelease="cos"
    if [[ -n "$(command -v lsb_release)" ]]; then
      linuxrelease=$(lsb_release -si)
    fi
    case "${linuxrelease}" in
      Ubuntu)
        log-wrap "InstallContainerdUbuntu" install-containerd-ubuntu
        ;;
      cos)
        log-wrap "InstallContainerdCOS" install-containerd-cos
        ;;
      *)
        echo "Installing containerd for linux release ${linuxrelease} not supported" >&2
        exit 2
        ;;
    esac
  fi

  # when custom containerd version is installed sourcing containerd_env.sh will add all tools like ctr to the PATH
  if [[ -e "/etc/profile.d/containerd_env.sh" ]]; then
   log-wrap 'SourceContainerdEnv' source "/etc/profile.d/containerd_env.sh"
  fi

  # Verify presence and print versions of ctr, containerd, runc
  if ! command -v ctr >/dev/null 2>&1; then
    echo "ERROR ctr not found. Aborting."
    exit 2
  fi
  ctr --version
  if ! command -v containerd >/dev/null 2>&1; then
    echo "ERROR containerd not found. Aborting."
    exit 2
  fi
  containerd --version
  if ! command -v runc >/dev/null 2>&1; then
    echo "ERROR runc not found. Aborting."
    exit 2
  fi
  runc --version
}

function ensure-container-runtime {
  case "${CONTAINER_RUNTIME_NAME:-containerd}" in
    containerd)
      ensure-containerd-runtime
      ;;
    #TODO: Add crio support
    *)
      echo "Unsupported container runtime (${CONTAINER_RUNTIME_NAME})." >&2
      exit 2
      ;;
  esac
}

# Downloads kubernetes binaries and kube-system manifest tarball, unpacks them,
# and places them into suitable directories. Files are placed in /home/kubernetes.
function install-kube-binary-config {
  cd "${KUBE_HOME}"
  local server_binary_tar_urls
  while IFS= read -r url; do
    server_binary_tar_urls+=("$url")
  done < <(split-commas "${SERVER_BINARY_TAR_URL}")
  local -r server_binary_tar="${server_binary_tar_urls[0]##*/}"
  if [[ -n "${SERVER_BINARY_TAR_HASH:-}" ]]; then
    local -r server_binary_tar_hash="${SERVER_BINARY_TAR_HASH}"
  else
    echo "Downloading binary release sha512 (not found in env)"
    log-wrap "DownloadServerBinarySHA" download-or-bust "" "${server_binary_tar_urls[@]/.tar.gz/.tar.gz.sha512}"
    local -r server_binary_tar_hash=$(cat "${server_binary_tar}.sha512")
  fi

  if is-preloaded "${server_binary_tar}" "${server_binary_tar_hash}"; then
    echo "${server_binary_tar} is preloaded."
  else
    echo "Downloading binary release tar"
    log-wrap "DownloadServerBinary" download-or-bust "${server_binary_tar_hash}" "${server_binary_tar_urls[@]}"
    log-wrap "UntarServerBinary" tar xzf "${KUBE_HOME}/${server_binary_tar}" -C "${KUBE_HOME}" --overwrite
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
    log-wrap "LoadDockerImages" load-docker-images
    mv "${src_dir}/kubelet" "${KUBE_BIN}"
    mv "${src_dir}/kubectl" "${KUBE_BIN}"

    # Some older images have LICENSES baked-in as a file. Presumably they will
    # have the directory baked-in eventually.
    rm -rf "${KUBE_HOME}"/LICENSES
    mv "${KUBE_HOME}/kubernetes/LICENSES" "${KUBE_HOME}"
    mv "${KUBE_HOME}/kubernetes/kubernetes-src.tar.gz" "${KUBE_HOME}"
  fi

  if [[ "${KUBERNETES_MASTER:-}" == "false" ]] && \
     [[ "${ENABLE_NODE_PROBLEM_DETECTOR:-}" == "standalone" ]]; then
    log-wrap "InstallNodeProblemDetector" install-node-problem-detector
  fi

  if [[ "${NETWORK_PROVIDER:-}" == "kubenet" ]] || \
     [[ "${NETWORK_PROVIDER:-}" == "cni" ]]; then
    log-wrap "InstallCNIBinaries" install-cni-binaries
  fi

  # Put kube-system pods manifests in ${KUBE_HOME}/kube-manifests/.
  log-wrap "InstallKubeManifests" install-kube-manifests
  chmod -R 755 "${KUBE_BIN}"

  # Install gci mounter related artifacts to allow mounting storage volumes in GCI
  log-wrap "InstallGCIMounterTools" install-gci-mounter-tools

  # Remount the Flexvolume directory with the "exec" option, if needed.
  if [[ "${REMOUNT_VOLUME_PLUGIN_DIR:-}" == "true" && -n "${VOLUME_PLUGIN_DIR:-}" ]]; then
    log-wrap "RemountFlexVolume" remount-flexvolume-directory "${VOLUME_PLUGIN_DIR}"
  fi

  # When ENABLE_AUTH_PROVIDER_GCP is set, following flags for out-of-tree credential provider for GCP
  # are presented to kubelet:
  # --image-credential-provider-config=${path-to-config}
  # --image-credential-provider-bin-dir=${path-to-auth-provider-binary}
  # Also, it is required that DisableKubeletCloudCredentialProviders
  # feature gate is set to true for kubelet to use external credential provider.
  if [[ "${ENABLE_AUTH_PROVIDER_GCP:-}" == "true" ]]; then
    # Install out-of-tree auth-provider-gcp binary to enable kubelet to dynamically
    # retrieve credentials for a container image registry.
    log-wrap "InstallCredentialProvider" install-auth-provider-gcp
  fi
  # Install crictl on each node.
  log-wrap "InstallCrictl" install-crictl

  # Clean up.
  rm -rf "${KUBE_HOME}/kubernetes"
  rm -f "${KUBE_HOME}/${server_binary_tar}"
  rm -f "${KUBE_HOME}/${server_binary_tar}.sha512"
}


# This function detects the platform/arch of the machine where the script runs,
# and sets the HOST_PLATFORM and HOST_ARCH environment variables accordingly.
# Callers can specify HOST_PLATFORM_OVERRIDE and HOST_ARCH_OVERRIDE to skip the detection.
# This function is adapted from the detect_client_info function in cluster/get-kube-binaries.sh
# and kube::util::host_os, kube::util::host_arch functions in hack/lib/util.sh
# This function should be synced with detect_host_info in ./configure-helper.sh
function detect_host_info() {
  HOST_PLATFORM=${HOST_PLATFORM_OVERRIDE:-"$(uname -s)"}
  case "${HOST_PLATFORM}" in
    Linux|linux)
      HOST_PLATFORM="linux"
      ;;
    *)
      echo "Unknown, unsupported platform: ${HOST_PLATFORM}." >&2
      echo "Supported platform(s): linux." >&2
      echo "Bailing out." >&2
      exit 2
  esac

  HOST_ARCH=${HOST_ARCH_OVERRIDE:-"$(uname -m)"}
  case "${HOST_ARCH}" in
    x86_64*|i?86_64*|amd64*)
      HOST_ARCH="amd64"
      ;;
    aHOST_arch64*|aarch64*|arm64*)
      HOST_ARCH="arm64"
      ;;
    *)
      echo "Unknown, unsupported architecture (${HOST_ARCH})." >&2
      echo "Supported architecture(s): amd64 and arm64." >&2
      echo "Bailing out." >&2
      exit 2
      ;;
  esac
}

# Retries a command forever with a delay between retries.
# Args:
#  $1    : delay between retries, in seconds.
#  $2... : the command to run.
function retry-forever {
  local -r delay="$1"
  shift 1

  until "$@"; do
    echo "== $* failed, retrying after ${delay}s"
    sleep "${delay}"
  done
}

# Initializes variables used by the log-* functions.
#
# get-metadata-value must be defined before calling this function.
#
# NOTE: this function is duplicated in configure-helper.sh, any changes here
# should be duplicated there as well.
function log-init {
  # Used by log-* functions.
  LOG_CLUSTER_ID=$(get-metadata-value 'instance/attributes/cluster-uid' 'get-metadata-value-error')
  LOG_INSTANCE_NAME=$(hostname)
  LOG_BOOT_ID=$(journalctl --list-boots | grep -E '^ *0' | awk '{print $2}')
  declare -Ag LOG_START_TIMES
  declare -ag LOG_TRAP_STACK

  LOG_STATUS_STARTED='STARTED'
  LOG_STATUS_COMPLETED='COMPLETED'
  LOG_STATUS_ERROR='ERROR'
}

# Sets an EXIT trap.
# Args:
#   $1:... : the trap command.
#
# NOTE: this function is duplicated in configure-helper.sh, any changes here
# should be duplicated there as well.
function log-trap-push {
  local t="${*:1}"
  LOG_TRAP_STACK+=("${t}")
  # shellcheck disable=2064
  trap "${t}" EXIT
}

# Removes and restores an EXIT trap.
#
# NOTE: this function is duplicated in configure-helper.sh, any changes here
# should be duplicated there as well.
function log-trap-pop {
  # Remove current trap.
  unset 'LOG_TRAP_STACK[-1]'

  # Restore previous trap.
  if [ ${#LOG_TRAP_STACK[@]} -ne 0 ]; then
    local t="${LOG_TRAP_STACK[-1]}"
    # shellcheck disable=2064
    trap "${t}" EXIT
  else
    # If no traps in stack, clear.
    trap EXIT
  fi
}

# Logs the end of a bootstrap step that errored.
# Args:
#  $1 : bootstrap step name.
#
# NOTE: this function is duplicated in configure-helper.sh, any changes here
# should be duplicated there as well.
function log-error {
  local bootstep="$1"

  log-proto "${bootstep}" "${LOG_STATUS_ERROR}" "encountered non-zero exit code"
}

# Wraps a command with bootstrap logging.
# Args:
#   $1    : bootstrap step name.
#   $2... : the command to run.
#
# NOTE: this function is duplicated in configure-helper.sh, any changes here
# should be duplicated there as well.
function log-wrap {
  local bootstep="$1"
  local command="${*:2}"

  log-trap-push "log-error ${bootstep}"
  log-proto "${bootstep}" "${LOG_STATUS_STARTED}"
  $command
  log-proto "${bootstep}" "${LOG_STATUS_COMPLETED}"
  log-trap-pop
}

# Logs a bootstrap step start. Prefer log-wrap.
# Args:
#   $1 : bootstrap step name.
#
# NOTE: this function is duplicated in configure-helper.sh, any changes here
# should be duplicated there as well.
function log-start {
  local bootstep="$1"

  log-trap-push "log-error ${bootstep}"
  log-proto "${bootstep}" "${LOG_STATUS_STARTED}"
}

# Logs a bootstrap step end. Prefer log-wrap.
# Args:
#   $1 : bootstrap step name.
#
# NOTE: this function is duplicated in configure-helper.sh, any changes here
# should be duplicated there as well.
function log-end {
  local bootstep="$1"

  log-proto "${bootstep}" "${LOG_STATUS_COMPLETED}"
  log-trap-pop
}

# Writes a log proto to stdout.
# Args:
#   $1: bootstrap step name.
#   $2: status. Either 'STARTED', 'COMPLETED', or 'ERROR'.
#   $3: optional status reason.
#
# NOTE: this function is duplicated in configure-helper.sh, any changes here
# should be duplicated there as well.
function log-proto {
  local bootstep="$1"
  local status="$2"
  local status_reason="${3:-}"

  # Get current time.
  local current_time
  current_time="$(date --utc '+%s.%N')"
  # ...formatted as UTC RFC 3339.
  local timestamp
  timestamp="$(date --utc --date="@${current_time}" '+%FT%T.%NZ')"

  # Calculate latency.
  local latency='null'
  if [ "${status}" == "${LOG_STATUS_STARTED}" ]; then
    LOG_START_TIMES["${bootstep}"]="${current_time}"
  else
    local start_time="${LOG_START_TIMES["${bootstep}"]}"
    unset 'LOG_START_TIMES['"${bootstep}"']'

    # Bash cannot do non-integer math, shell out to awk.
    latency="$(echo "${current_time} ${start_time}" | awk '{print $1 - $2}')s"

    # The default latency is null which cannot be wrapped as a string so we must
    # do it here instead of the printf.
    latency="\"${latency}\""
  fi

  printf '[cloud.kubernetes.monitoring.proto.SerialportLog] {"cluster_hash":"%s","vm_instance_name":"%s","boot_id":"%s","timestamp":"%s","bootstrap_status":{"step_name":"%s","status":"%s","status_reason":"%s","latency":%s}}\n' \
  "${LOG_CLUSTER_ID}" "${LOG_INSTANCE_NAME}" "${LOG_BOOT_ID}" "${timestamp}" "${bootstep}" "${status}" "${status_reason}" "${latency}"
}

######### Main Function ##########
log-init
log-start 'ConfigureMain'
echo "Start to install kubernetes files"
log-wrap 'DetectHostInfo' detect_host_info

# if install fails, message-of-the-day (motd) will warn at login shell
log-wrap 'SetBrokenMotd' set-broken-motd

KUBE_HOME="/home/kubernetes"
KUBE_BIN="${KUBE_HOME}/bin"

# download and source kube-env
log-wrap 'DownloadKubeEnv' download-kube-env
log-wrap 'SourceKubeEnv' source "${KUBE_HOME}/kube-env"

log-wrap 'DownloadKubeletConfig' download-kubelet-config "${KUBE_HOME}/kubelet-config.yaml"

# master certs
if [[ "${KUBERNETES_MASTER:-}" == "true" ]]; then
  log-wrap 'DownloadKubeMasterCerts' download-kube-master-certs
fi

# ensure chosen container runtime is present
log-wrap 'EnsureContainerRuntime' ensure-container-runtime

# binaries and kube-system manifests
log-wrap 'InstallKubeBinaryConfig' install-kube-binary-config

echo "Done for installing kubernetes files"
log-end 'ConfigureMain'
