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

# Builds a docker image that generates ssl certificates/keys/tokens required by kubernetes

set -o errexit
set -o nounset
set -o pipefail

IMAGE_REPO=${IMAGE_REPO:-mesosphere/kubernetes-mesos-keygen}
IMAGE_TAG=${IMAGE_TAG:-latest}

script_dir=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
common_bin_path=$(cd ${script_dir}/../common/bin && pwd -P)
KUBE_ROOT=$(cd ${script_dir}/../../../.. && pwd -P)

source "${common_bin_path}/util-temp-dir.sh"

cd "${KUBE_ROOT}"

function build_image {
  local -r workspace="$(pwd)"

  echo "Copying files to workspace"

  # binaries & scripts
  mkdir -p "${workspace}/bin"
  cp -a "${common_bin_path}/"* "${workspace}/bin/"
  cp -a "${script_dir}/bin/"* "${workspace}/bin/"

  # docker
  cp -a "${script_dir}/Dockerfile" "${workspace}/"

  echo "Building docker image ${IMAGE_REPO}:${IMAGE_TAG}"
  set -o xtrace
  docker build -t ${IMAGE_REPO}:${IMAGE_TAG} "$@" .
  set +o xtrace
  echo "Built docker image ${IMAGE_REPO}:${IMAGE_TAG}"
}

cluster::mesos::docker::run_in_temp_dir 'k8sm-keygen' 'build_image'
