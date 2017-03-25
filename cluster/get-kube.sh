#!/usr/bin/env bash

# Copyright 2014 The Kubernetes Authors.
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

# Bring up a Kubernetes cluster.
# Usage:
#   wget -q -O - https://get.k8s.io | bash
# or
#   curl -fsSL https://get.k8s.io | bash
#
# Advanced options
#  Set KUBERNETES_PROVIDER to choose between different providers:
#  Google Compute Engine [default]
#   * export KUBERNETES_PROVIDER=gce; wget -q -O - https://get.k8s.io | bash
#  Google Container Engine
#   * export KUBERNETES_PROVIDER=gke; wget -q -O - https://get.k8s.io | bash
#  Amazon EC2
#   * export KUBERNETES_PROVIDER=aws; wget -q -O - https://get.k8s.io | bash
#  Libvirt (with CoreOS as a guest operating system)
#   * export KUBERNETES_PROVIDER=libvirt-coreos; wget -q -O - https://get.k8s.io | bash
#  Microsoft Azure
#   * export KUBERNETES_PROVIDER=azure-legacy; wget -q -O - https://get.k8s.io | bash
#  Vagrant (local virtual machines)
#   * export KUBERNETES_PROVIDER=vagrant; wget -q -O - https://get.k8s.io | bash
#  VMWare Photon Controller
#   * export KUBERNETES_PROVIDER=photon-controller; wget -q -O - https://get.k8s.io | bash
#  Rackspace
#   * export KUBERNETES_PROVIDER=rackspace; wget -q -O - https://get.k8s.io | bash
#  OpenStack-Heat
#   * export KUBERNETES_PROVIDER=openstack-heat; wget -q -O - https://get.k8s.io | bash
#
#  Set KUBERNETES_RELEASE to choose a specific release instead of the current
#    stable release, (e.g. 'v1.3.7').
#    See https://github.com/kubernetes/kubernetes/releases for release options.
#  Set KUBERNETES_RELEASE_URL to choose where to download binaries from.
#    (Defaults to https://storage.googleapis.com/kubernetes-release/release).
#
#  Set KUBERNETES_SERVER_ARCH to choose the server (Kubernetes cluster)
#  architecture to download:
#    * amd64 [default]
#    * arm
#    * arm64
#
#  Set KUBERNETES_SKIP_DOWNLOAD to skip downloading a release.
#  Set KUBERNETES_SKIP_CONFIRM to skip the installation confirmation prompt.
#  Set KUBERNETES_SKIP_CREATE_CLUSTER to skip starting a cluster.

set -o errexit
set -o nounset
set -o pipefail

# If KUBERNETES_RELEASE_URL is overridden but KUBERNETES_CI_RELEASE_URL is not then set KUBERNETES_CI_RELEASE_URL to KUBERNETES_RELEASE_URL.
KUBERNETES_CI_RELEASE_URL="${KUBERNETES_CI_RELEASE_URL:-${KUBERNETES_RELEASE_URL:-https://dl.k8s.io/ci}}"
KUBERNETES_RELEASE_URL="${KUBERNETES_RELEASE_URL:-https://dl.k8s.io}"

KUBE_RELEASE_VERSION_REGEX="^v(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)(-(beta|alpha|rc)\\.(0|[1-9][0-9]*))?$"
KUBE_CI_VERSION_REGEX="^v(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)-(beta|alpha|rc)\\.(0|[1-9][0-9]*)(\\.(0|[1-9][0-9]*)\\+[-0-9a-z]*)?$"

# Sets KUBE_VERSION variable if an explicit version number was provided (e.g. "v1.0.6",
# "v1.2.0-alpha.1.881+376438b69c7612") or resolves the "published" version
# <path>/<version> (e.g. "release/stable",' "ci/latest-1") by reading from GCS.
#
# See the docs on getting builds for more information about version
# publication.
#
# Args:
#   $1 version string from command line
# Vars set:
#   KUBE_VERSION
function set_binary_version() {
  if [[ "${1}" =~ "/" ]]; then
    export KUBE_VERSION=$(curl -fL "https://dl.k8s.io/${1}.txt")
  else
    export KUBE_VERSION=${1}
  fi
}

# Use the script from inside the Kubernetes tarball to fetch the client and
# server binaries (if not included in kubernetes.tar.gz).
function download_kube_binaries {
  (
    cd kubernetes
    if [[ -x ./cluster/get-kube-binaries.sh ]]; then
      ./cluster/get-kube-binaries.sh
    fi
  )
}

function create_cluster {
  if [[ -n "${KUBERNETES_SKIP_CREATE_CLUSTER-}" ]]; then
    exit 0
  fi
  echo "Creating a kubernetes on ${KUBERNETES_PROVIDER:-gce}..."
  (
    cd kubernetes
    ./cluster/kube-up.sh
    echo "Kubernetes binaries at ${PWD}/cluster/"
    if [[ ":$PATH:" != *":${PWD}/cluster:"* ]]; then
      echo "You may want to add this directory to your PATH in \$HOME/.profile"
    fi

    echo "Installation successful!"
  )
}

if [[ -n "${KUBERNETES_SKIP_DOWNLOAD-}" ]]; then
  create_cluster
  exit 0
fi

if [[ -d "./kubernetes" ]]; then
  if [[ -z "${KUBERNETES_SKIP_CONFIRM-}" ]]; then
    echo "'kubernetes' directory already exist. Should we skip download step and start to create cluster based on it? [Y]/n"
    read confirm
    if [[ ! "${confirm}" =~ ^[nN]$ ]]; then
      echo "Skipping download step."
      create_cluster
      exit 0
    fi
  fi
fi

# TODO: remove client checks once kubernetes.tar.gz no longer includes client
# binaries by default.
kernel=$(uname -s)
case "${kernel}" in
  Darwin)
    platform="darwin"
    ;;
  Linux)
    platform="linux"
    ;;
  *)
    echo "Unknown, unsupported platform: ${kernel}." >&2
    echo "Supported platforms: Linux, Darwin." >&2
    echo "Bailing out." >&2
    exit 2
esac

machine=$(uname -m)
case "${machine}" in
  x86_64*|i?86_64*|amd64*)
    arch="amd64"
    ;;
  aarch64*|arm64*)
    arch="arm64"
    ;;
  arm*)
    arch="arm"
    ;;
  i?86*)
    arch="386"
    ;;
  *)
    echo "Unknown, unsupported architecture (${machine})." >&2
    echo "Supported architectures x86_64, i686, arm, arm64." >&2
    echo "Bailing out." >&2
    exit 3
    ;;
esac

file=kubernetes.tar.gz
release=${KUBERNETES_RELEASE:-"release/stable"}

# Validate Kubernetes release version.
# Translate a published version <bucket>/<version> (e.g. "release/stable") to version number.
set_binary_version "${release}"
if [[ ${KUBE_VERSION} =~ ${KUBE_RELEASE_VERSION_REGEX} ]]; then
  release_url="${KUBERNETES_RELEASE_URL}/${KUBE_VERSION}/${file}"
elif [[ ${KUBE_VERSION} =~ ${KUBE_CI_VERSION_REGEX} ]]; then
  release_url="${KUBERNETES_CI_RELEASE_URL}/${KUBE_VERSION}/${file}"
else
  echo "Version doesn't match regexp" >&2
  exit 1
fi


need_download=true
if [[ -r "${PWD}/${file}" ]]; then
  downloaded_version=$(tar -xzOf "${PWD}/${file}" kubernetes/version 2>/dev/null || true)
  echo "Found preexisting ${file}, release ${downloaded_version}"
  if [[ "${downloaded_version}" == "${KUBE_VERSION}" ]]; then
    echo "Using preexisting kubernetes.tar.gz"
    need_download=false
  fi
fi

if "${need_download}"; then
  echo "Downloading kubernetes release ${KUBE_VERSION}"
  echo "  from ${release_url}"
  echo "  to ${PWD}/${file}"
fi

if [[ -e "${PWD}/kubernetes" ]]; then
  # Let's try not to accidentally nuke something that isn't a kubernetes
  # release dir.
  if [[ ! -f "${PWD}/kubernetes/version" ]]; then
    echo "${PWD}/kubernetes exists but does not look like a Kubernetes release."
    echo "Aborting!"
    exit 5
  fi
  echo "Will also delete preexisting 'kubernetes' directory."
fi

if [[ -z "${KUBERNETES_SKIP_CONFIRM-}" ]]; then
  echo "Is this ok? [Y]/n"
  read confirm
  if [[ "${confirm}" =~ ^[nN]$ ]]; then
    echo "Aborting."
    exit 0
  fi
fi

if "${need_download}"; then
  if [[ $(which curl) ]]; then
    curl -fL --retry 3 --keepalive-time 2 "${release_url}" -o "${file}"
  elif [[ $(which wget) ]]; then
    wget "${release_url}"
  else
    echo "Couldn't find curl or wget.  Bailing out."
    exit 1
  fi
fi

echo "Unpacking kubernetes release ${KUBE_VERSION}"
rm -rf "${PWD}/kubernetes"
tar -xzf ${file}

download_kube_binaries
create_cluster
