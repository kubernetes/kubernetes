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

# Builds a docker image that contains the kubernetes-mesos binaries.

set -o errexit
set -o nounset
set -o pipefail

IMAGE_REPO=${IMAGE_REPO:-mesosphere-local/kubernetes-mesos-slave-dind}
IMAGE_TAG=${IMAGE_TAG:-latest}

script_dir=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
KUBE_ROOT=$(cd ${script_dir}/../../../.. && pwd -P)

# download nsenter
mkdir -p "${script_dir}/downloads"
docker run --rm -v "${script_dir}/downloads:/target" jpetazzo/nsenter

# create temp workspace to place compiled binaries with image-specific scripts
# create temp workspace dir in KUBE_ROOT to avoid permission issues of TMPDIR on mac os x
cd "${KUBE_ROOT}"
workspace=$(env TMPDIR=$PWD mktemp -d -t "k8sm-workspace-XXXXXX")
echo "Workspace created: ${workspace}"

cleanup() {
  rm -rf "${workspace}"
  echo "Workspace deleted: ${workspace}"
}
trap 'cleanup' EXIT

# setup workspace to mirror script dir (dockerfile expects /bin & /opt)
echo "Copying files to workspace"
cp "${script_dir}/Dockerfile" "${workspace}/"
cp -a "${script_dir}/downloads" "${workspace}/"

cd "${workspace}"

# build docker image
echo "Building docker image ${IMAGE_REPO}:${IMAGE_TAG}"
set -o xtrace
docker build -t ${IMAGE_REPO}:${IMAGE_TAG} "$@" .
set +o xtrace
echo "Built docker image ${IMAGE_REPO}:${IMAGE_TAG}"
