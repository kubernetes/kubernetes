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

set -o errexit
set -o nounset
set -o pipefail

IMAGE_REPO=${IMAGE_REPO:-mesosphere/kubernetes-mesos-test}
IMAGE_TAG=${IMAGE_TAG:-latest}

script_dir=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
common_bin_path=$(cd ${script_dir}/../common/bin && pwd -P)
KUBE_ROOT=$(cd ${script_dir}/../../../.. && pwd -P)

cd "${KUBE_ROOT}"

# create temp workspace to place common scripts with image-specific scripts
# create temp workspace dir in KUBE_ROOT to avoid permission issues of TMPDIR on mac os x
workspace=$(env TMPDIR=$PWD mktemp -d -t "k8sm-test-workspace-XXXXXX")
echo "Workspace created: ${workspace}"

cleanup() {
  rm -rf "${workspace}"
  echo "Workspace deleted: ${workspace}"
}
trap 'cleanup' EXIT

# setup workspace to mirror script dir (dockerfile expects /bin)
set -x
echo "Copying files to workspace"

# binaries & scripts
mkdir -p "${workspace}/bin"
cp -a "${common_bin_path}/"* "${workspace}/bin/"
cp -a "${script_dir}/bin/"* "${workspace}/bin/"

# docker
cp -a "${script_dir}/Dockerfile" "${workspace}/"

cd "${workspace}"

echo "Building docker image ${IMAGE_REPO}:${IMAGE_TAG}"
set -o xtrace
docker build -t ${IMAGE_REPO}:${IMAGE_TAG} "$@" .
set +o xtrace
echo "Built docker image ${IMAGE_REPO}:${IMAGE_TAG}"
