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

# Builds a docker image that contains the kubernetes-mesos binaries.

set -o errexit
set -o nounset
set -o pipefail

IMAGE_REPO=${IMAGE_REPO:-mesosphere/kubernetes-mesos}
IMAGE_TAG=${IMAGE_TAG:-latest}

script_dir=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
KUBE_ROOT=$(cd ${script_dir}/../../../.. && pwd -P)

# Find a platform specific binary, whether it was cross compiled, locally built, or downloaded.
find-binary() {
  local lookfor="${1}"
  local platform="${2}"
  local locations=(
    "${KUBE_ROOT}/_output/dockerized/bin/${platform}/${lookfor}"
    "${KUBE_ROOT}/_output/local/bin/${platform}/${lookfor}"
    "${KUBE_ROOT}/platforms/${platform}/${lookfor}"
  )
  local bin=$( (ls -t "${locations[@]}" 2>/dev/null || true) | head -1 )
  echo -n "${bin}"
}

km_path=$(find-binary km linux/amd64)
if [ -z "$km_path" ]; then
  km_path=$(find-binary km darwin/amd64)
  if [ -z "$km_path" ]; then
    echo "Failed to find km binary" 1>&2
    exit 1
  fi
fi
kube_bin_path=$(dirname ${km_path})
common_bin_path=$(cd ${script_dir}/../common/bin && pwd -P)

# download nsenter and socat
overlay_dir=${MESOS_DOCKER_OVERLAY_DIR:-${script_dir}/overlay}
mkdir -p "${overlay_dir}"
docker run --rm -v "${overlay_dir}:/target" jpetazzo/nsenter
docker run --rm -v "${overlay_dir}:/target" mesosphere/kubernetes-socat

cd "${KUBE_ROOT}"

# create temp workspace to place compiled binaries with image-specific scripts
# create temp workspace dir in KUBE_ROOT to avoid permission issues of TMPDIR on mac os x
workspace=$(env TMPDIR=$PWD mktemp -d -t "k8sm-workspace-XXXXXX")
echo "Workspace created: ${workspace}"

cleanup() {
  rm -rf "${workspace}"
  rm -f "${overlay_dir}/*"
  echo "Workspace deleted: ${workspace}"
}
trap 'cleanup' EXIT

# setup workspace to mirror script dir (dockerfile expects /bin & /opt)
echo "Copying files to workspace"

# binaries & scripts
mkdir -p "${workspace}/bin"

#cp "${script_dir}/bin/"* "${workspace}/bin/"
cp "${common_bin_path}/"* "${workspace}/bin/"
cp "${kube_bin_path}/km" "${workspace}/bin/"

# config
mkdir -p "${workspace}/opt"
cp "${script_dir}/opt/"* "${workspace}/opt/"

# package up the sandbox overay
mkdir -p "${workspace}/overlay/bin"
cp -a "${overlay_dir}/nsenter" "${workspace}/overlay/bin"
cp -a "${overlay_dir}/socat" "${workspace}/overlay/bin"
chmod +x "${workspace}/overlay/bin/"*
cd "${workspace}/overlay" && tar -czvf "${workspace}/opt/sandbox-overlay.tar.gz" . && cd -

# docker
cp "${script_dir}/Dockerfile" "${workspace}/"

cd "${workspace}"

# build docker image
echo "Building docker image ${IMAGE_REPO}:${IMAGE_TAG}"
set -o xtrace
docker build -t ${IMAGE_REPO}:${IMAGE_TAG} "$@" .
set +o xtrace
echo "Built docker image ${IMAGE_REPO}:${IMAGE_TAG}"
