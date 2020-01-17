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

# This script downloads and installs the Kubernetes client and server
# (and optionally test) binaries,
# It is intended to be called from an extracted Kubernetes release tarball.
#
# We automatically choose the correct client binaries to download.
#
# Options:
#  Set KUBERNETES_SERVER_ARCH to choose the server (Kubernetes cluster)
#  architecture to download:
#    * amd64 [default]
#    * arm
#    * arm64
#    * ppc64le
#    * s390x
#
#  Set KUBERNETES_CLIENT_OS to choose the client OS to download:
#    * current OS [default]
#    * linux
#    * darwin
#    * windows
#
#  Set KUBERNETES_CLIENT_ARCH to choose the client architecture to download:
#    * current architecture [default]
#    * amd64
#    * arm
#    * arm64
#    * ppc64le
#    * s390x
#    * windows
#
#  Set KUBERNETES_SKIP_CONFIRM to skip the installation confirmation prompt.
#  Set KUBERNETES_RELEASE_URL to choose where to download binaries from.
#    (Defaults to https://storage.googleapis.com/kubernetes-release/release).
#  Set KUBERNETES_DOWNLOAD_TESTS to additionally download and extract the test
#    binaries tarball.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

KUBERNETES_RELEASE_URL="${KUBERNETES_RELEASE_URL:-https://dl.k8s.io}"

function detect_kube_release() {
  if [[ -n "${KUBE_VERSION:-}" ]]; then
    return 0  # Allow caller to explicitly set version
  fi

  if [[ ! -e "${KUBE_ROOT}/version" ]]; then
    echo "Can't determine Kubernetes release." >&2
    echo "${BASH_SOURCE[0]} should only be run from a prebuilt Kubernetes release." >&2
    echo "Did you mean to use get-kube.sh instead?" >&2
    exit 1
  fi

  KUBE_VERSION=$(cat "${KUBE_ROOT}/version")
}

function detect_client_info() {
  if [ -n "${KUBERNETES_CLIENT_OS-}" ]; then
    CLIENT_PLATFORM="${KUBERNETES_CLIENT_OS}"
  else
    local kernel
    kernel="$(uname -s)"
    case "${kernel}" in
      Darwin)
        CLIENT_PLATFORM="darwin"
        ;;
      Linux)
        CLIENT_PLATFORM="linux"
        ;;
      *)
        echo "Unknown, unsupported platform: ${kernel}." >&2
        echo "Supported platforms: Linux, Darwin." >&2
        echo "Bailing out." >&2
        exit 2
    esac
  fi

  if [ -n "${KUBERNETES_CLIENT_ARCH-}" ]; then
    CLIENT_ARCH="${KUBERNETES_CLIENT_ARCH}"
  else
    # TODO: migrate the kube::util::host_platform function out of hack/lib and
    # use it here.
    local machine
    machine="$(uname -m)"
    case "${machine}" in
      x86_64*|i?86_64*|amd64*)
        CLIENT_ARCH="amd64"
        ;;
      aarch64*|arm64*)
        CLIENT_ARCH="arm64"
        ;;
      arm*)
        CLIENT_ARCH="arm"
        ;;
      i?86*)
        CLIENT_ARCH="386"
        ;;
      s390x*)
        CLIENT_ARCH="s390x"
        ;;
      *)
        echo "Unknown, unsupported architecture (${machine})." >&2
        echo "Supported architectures x86_64, i686, arm, arm64, s390x." >&2
        echo "Bailing out." >&2
        exit 3
        ;;
    esac
  fi
}

function md5sum_file() {
  if which md5 >/dev/null 2>&1; then
    md5 -q "$1"
  else
    md5sum "$1" | awk '{ print $1 }'
  fi
}

function sha1sum_file() {
  if which sha1sum >/dev/null 2>&1; then
    sha1sum "$1" | awk '{ print $1 }'
  else
    shasum -a1 "$1" | awk '{ print $1 }'
  fi
}

# Get default service account credentials of the VM.
GCE_METADATA_INTERNAL="http://metadata.google.internal/computeMetadata/v1/instance"
function get-credentials {
  curl "${GCE_METADATA_INTERNAL}/service-accounts/default/token" -H "Metadata-Flavor: Google" -s | python -c \
    'import sys; import json; print(json.loads(sys.stdin.read())["access_token"])'
}

function valid-storage-scope {
  curl "${GCE_METADATA_INTERNAL}/service-accounts/default/scopes" -H "Metadata-Flavor: Google" -s | grep -E "auth/devstorage|auth/cloud-platform"
}

function download_tarball() {
  local -r download_path="$1"
  local -r file="$2"
  local trace_on="off"
  if [[ -o xtrace ]]; then 
    trace_on="on"
    set +x
  fi
  url="${DOWNLOAD_URL_PREFIX}/${file}"
  mkdir -p "${download_path}"
  if [[ $(which curl) ]]; then
    # if the url belongs to GCS API we should use oauth2_token in the headers
    curl_headers=""
    if { [[ "${KUBERNETES_PROVIDER:-gce}" == "gce" ]] || [[ "${KUBERNETES_PROVIDER}" == "gke" ]] ; } &&
       [[ "$url" =~ ^https://storage.googleapis.com.* ]] && valid-storage-scope ; then
      curl_headers="Authorization: Bearer $(get-credentials)"
    fi
    curl ${curl_headers:+-H "${curl_headers}"} -fL --retry 3 --keepalive-time 2 "${url}" -o "${download_path}/${file}"
  elif [[ $(which wget) ]]; then
    wget "${url}" -O "${download_path}/${file}"
  else
    echo "Couldn't find curl or wget.  Bailing out." >&2
    exit 4
  fi
  echo
  local md5sum sha1sum
  md5sum=$(md5sum_file "${download_path}/${file}")
  echo "md5sum(${file})=${md5sum}"
  sha1sum=$(sha1sum_file "${download_path}/${file}")
  echo "sha1sum(${file})=${sha1sum}"
  echo
  # TODO: add actual verification
  if [[ "${trace_on}" == "on" ]]; then
    set -x
  fi
}

function extract_arch_tarball() {
  local -r tarfile="$1"
  local -r platform="$2"
  local -r arch="$3"

  platforms_dir="${KUBE_ROOT}/platforms/${platform}/${arch}"
  echo "Extracting ${tarfile} into ${platforms_dir}"
  mkdir -p "${platforms_dir}"
  # Tarball looks like kubernetes/{client,server,test}/bin/BINARY"
  tar -xzf "${tarfile}" --strip-components 3 -C "${platforms_dir}"
}

detect_kube_release
DOWNLOAD_URL_PREFIX="${KUBERNETES_RELEASE_URL}/${KUBE_VERSION}"

SERVER_PLATFORM="linux"
SERVER_ARCH="${KUBERNETES_SERVER_ARCH:-amd64}"
SERVER_TAR="kubernetes-server-${SERVER_PLATFORM}-${SERVER_ARCH}.tar.gz"
if [[ -n "${KUBERNETES_NODE_PLATFORM-}" || -n "${KUBERNETES_NODE_ARCH-}" ]]; then
  NODE_PLATFORM="${KUBERNETES_NODE_PLATFORM:-${SERVER_PLATFORM}}"
  NODE_ARCH="${KUBERNETES_NODE_ARCH:-${SERVER_ARCH}}"
  NODE_TAR="kubernetes-node-${NODE_PLATFORM}-${NODE_ARCH}.tar.gz"
fi

detect_client_info
CLIENT_TAR="kubernetes-client-${CLIENT_PLATFORM}-${CLIENT_ARCH}.tar.gz"

echo "Kubernetes release: ${KUBE_VERSION}"
echo "Server: ${SERVER_PLATFORM}/${SERVER_ARCH}  (to override, set KUBERNETES_SERVER_ARCH)"
printf "Client: %s/%s" "${CLIENT_PLATFORM}" "${CLIENT_ARCH}"
if [ -z "${KUBERNETES_CLIENT_OS-}" ] && [ -z "${KUBERNETES_CLIENT_ARCH-}" ]; then
  printf "  (autodetected)"
fi
echo "  (to override, set KUBERNETES_CLIENT_OS and/or KUBERNETES_CLIENT_ARCH)"
echo

echo "Will download ${SERVER_TAR} from ${DOWNLOAD_URL_PREFIX}"
echo "Will download and extract ${CLIENT_TAR} from ${DOWNLOAD_URL_PREFIX}"

DOWNLOAD_NODE_TAR=false
if [[ -n "${NODE_TAR:-}" ]]; then
  DOWNLOAD_NODE_TAR=true
  echo "Will download and extract ${NODE_TAR} from ${DOWNLOAD_URL_PREFIX}"
fi

DOWNLOAD_TESTS_TAR=false
if [[ -n "${KUBERNETES_DOWNLOAD_TESTS-}" ]]; then
  DOWNLOAD_TESTS_TAR=true
  echo "Will download and extract kubernetes-test tarball(s) from ${DOWNLOAD_URL_PREFIX}"
fi

if [[ -z "${KUBERNETES_SKIP_CONFIRM-}" ]]; then
  echo "Is this ok? [Y]/n"
  read -r confirm
  if [[ "${confirm}" =~ ^[nN]$ ]]; then
    echo "Aborting."
    exit 1
  fi
fi

download_tarball "${KUBE_ROOT}/server" "${SERVER_TAR}"

if "${DOWNLOAD_NODE_TAR}"; then
  download_tarball "${KUBE_ROOT}/node" "${NODE_TAR}"
fi

download_tarball "${KUBE_ROOT}/client" "${CLIENT_TAR}"
extract_arch_tarball "${KUBE_ROOT}/client/${CLIENT_TAR}" "${CLIENT_PLATFORM}" "${CLIENT_ARCH}"
ln -s "${KUBE_ROOT}/platforms/${CLIENT_PLATFORM}/${CLIENT_ARCH}" "${KUBE_ROOT}/client/bin"
echo "Add '${KUBE_ROOT}/client/bin' to your PATH to use newly-installed binaries."

if "${DOWNLOAD_TESTS_TAR}"; then
  TESTS_PORTABLE_TAR="kubernetes-test-portable.tar.gz"
  download_tarball "${KUBE_ROOT}/test" "${TESTS_PORTABLE_TAR}" || true
  if [[ -f "${KUBE_ROOT}/test/${TESTS_PORTABLE_TAR}" ]]; then
    echo "Extracting ${TESTS_PORTABLE_TAR} into ${KUBE_ROOT}"
    # Strip leading "kubernetes/"
    tar -xzf "${KUBE_ROOT}/test/${TESTS_PORTABLE_TAR}" --strip-components 1 -C "${KUBE_ROOT}"

    # Next, download platform-specific test tarballs for all relevant platforms
    TEST_PLATFORM_TUPLES=(
      "${CLIENT_PLATFORM}/${CLIENT_ARCH}"
      "${SERVER_PLATFORM}/${SERVER_ARCH}"
      )
    if [[ -n "${NODE_PLATFORM:-}" && -n "${NODE_ARCH:-}" ]]; then
      TEST_PLATFORM_TUPLES+=("${NODE_PLATFORM}/${NODE_ARCH}")
    fi
    # Loop over only the unique tuples
    for TUPLE in $(printf "%s\n" "${TEST_PLATFORM_TUPLES[@]}" | sort -u); do
        OS=$(echo "${TUPLE}" | cut -d/ -f1)
        ARCH=$(echo "${TUPLE}" | cut -d/ -f2)
        TEST_PLATFORM_TAR="kubernetes-test-${OS}-${ARCH}.tar.gz"
        download_tarball "${KUBE_ROOT}/test" "${TEST_PLATFORM_TAR}"
        extract_arch_tarball "${KUBE_ROOT}/test/${TEST_PLATFORM_TAR}" "${OS}" "${ARCH}"
    done
  else
    echo "Failed to download portable test tarball, falling back to mondo test tarball."
    TESTS_MONDO_TAR="kubernetes-test.tar.gz"
    download_tarball "${KUBE_ROOT}/test" "${TESTS_MONDO_TAR}"
    echo "Extracting ${TESTS_MONDO_TAR} into ${KUBE_ROOT}"
    # Strip leading "kubernetes/"
    tar -xzf "${KUBE_ROOT}/test/${TESTS_MONDO_TAR}" --strip-components 1 -C "${KUBE_ROOT}"
  fi
fi
