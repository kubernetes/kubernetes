#!/usr/bin/env bash

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

# Download the flannel, etcd, docker, bridge-utils and K8s binaries automatically
# and store into binaries directory.
# Run as sudoers only

# author @kevin-wangzefeng

set -o errexit
set -o nounset
set -o pipefail

readonly ROOT=$(dirname "${BASH_SOURCE[0]}")
source "${ROOT}/config-build.sh"

# ensure $RELEASES_DIR is an absolute file path
mkdir -p "${RELEASES_DIR}"
RELEASES_DIR=$(cd "${RELEASES_DIR}"; pwd)

# get absolute file path of binaries
BINARY_DIR=$(cd "${ROOT}"; pwd)/binaries

function clean-up() {
  rm -rf "${RELEASES_DIR}"
  rm -rf "${BINARY_DIR}"
}

function download-releases() {
  rm -rf "${RELEASES_DIR}"
  mkdir -p "${RELEASES_DIR}"

  echo "Download flannel release v${FLANNEL_VERSION} ..."
  curl -L "${FLANNEL_DOWNLOAD_URL}" -o "${RELEASES_DIR}/flannel.tar.gz"

  echo "Download etcd release v${ETCD_VERSION} ..."
  curl -L "${ETCD_DOWNLOAD_URL}" -o "${RELEASES_DIR}/etcd.tar.gz"

  echo "Download kubernetes release v${K8S_VERSION} ..."
  curl -L "${K8S_CLIENT_DOWNLOAD_URL}" -o "${RELEASES_DIR}/kubernetes-client-linux-amd64.tar.gz"
  curl -L "${K8S_SERVER_DOWNLOAD_URL}" -o "${RELEASES_DIR}/kubernetes-server-linux-amd64.tar.gz"

  echo "Download docker release v${DOCKER_VERSION} ..."
  curl -L "${DOCKER_DOWNLOAD_URL}" -o "${RELEASES_DIR}/docker.tar.gz"
}

function unpack-releases() {
  rm -rf "${BINARY_DIR}"
  mkdir -p "${BINARY_DIR}/master/bin"
  mkdir -p "${BINARY_DIR}/node/bin"

  # flannel
  if [[ -f "${RELEASES_DIR}/flannel.tar.gz" ]] ; then
    tar xzf "${RELEASES_DIR}/flannel.tar.gz" -C "${RELEASES_DIR}"
    cp "${RELEASES_DIR}/flanneld" "${BINARY_DIR}/master/bin"
    cp "${RELEASES_DIR}/flanneld" "${BINARY_DIR}/node/bin"
  fi

  # etcd
  if [[ -f "${RELEASES_DIR}/etcd.tar.gz" ]] ; then
    tar xzf "${RELEASES_DIR}/etcd.tar.gz" -C "${RELEASES_DIR}"
    ETCD="etcd-v${ETCD_VERSION}-linux-amd64"
    cp "${RELEASES_DIR}/${ETCD}/etcd" \
       "${RELEASES_DIR}/${ETCD}/etcdctl" "${BINARY_DIR}/master/bin"
    cp "${RELEASES_DIR}/${ETCD}/etcd" \
       "${RELEASES_DIR}/${ETCD}/etcdctl" "${BINARY_DIR}/node/bin"
  fi

  # k8s
  if [[ -f "${RELEASES_DIR}/kubernetes-client-linux-amd64.tar.gz" ]] ; then
    tar xzf "${RELEASES_DIR}/kubernetes-client-linux-amd64.tar.gz" -C "${RELEASES_DIR}"
    cp "${RELEASES_DIR}/kubernetes/client/bin/kubectl" "${BINARY_DIR}"
  fi

  if [[ -f "${RELEASES_DIR}/kubernetes-server-linux-amd64.tar.gz" ]] ; then
    tar xzf "${RELEASES_DIR}/kubernetes-server-linux-amd64.tar.gz" -C "${RELEASES_DIR}"
    cp "${RELEASES_DIR}/kubernetes/server/bin/kube-apiserver" \
       "${RELEASES_DIR}/kubernetes/server/bin/kube-controller-manager" \
       "${RELEASES_DIR}/kubernetes/server/bin/kube-scheduler" "${BINARY_DIR}/master/bin"
    cp "${RELEASES_DIR}/kubernetes/server/bin/kubelet" \
       "${RELEASES_DIR}/kubernetes/server/bin/kube-proxy" "${BINARY_DIR}/node/bin"
  fi

  # docker
  if [[ -f "${RELEASES_DIR}/docker.tar.gz" ]]; then
    tar xzf "${RELEASES_DIR}/docker.tar.gz" -C "${RELEASES_DIR}"

    cp "${RELEASES_DIR}/docker/docker*" "${BINARY_DIR}/node/bin"
  fi

  chmod -R +x "${BINARY_DIR}"
  echo "Done! All binaries are stored in ${BINARY_DIR}"
}

function parse-opt() {
  local opt=${1-}

  case $opt in
    download)
      download-releases
      ;;
    unpack)
      unpack-releases
      ;;
    clean)
      clean-up
      ;;
    all)
      download-releases
      unpack-releases
      ;;
    *)
      echo "Usage: "
      echo "   build.sh <command>"
      echo "Commands:"
      echo "   clean      Clean up downloaded releases and unpacked binaries."
      echo "   download   Download releases to \"${RELEASES_DIR}\"."
      echo "   unpack     Unpack releases downloaded in \"${RELEASES_DIR}\", and copy binaries to \"${BINARY_DIR}\"."
      echo "   all        Download releases and unpack them."
      ;;
  esac
}

parse-opt "${@}"
