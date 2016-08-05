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

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/build/common.sh"

kube::golang::setup_env

function prereqs() {
  kube::log::status "Verifying Prerequisites...."
  kube::build::ensure_docker_in_path || return 1
  if kube::build::is_osx; then
      kube::build::docker_available_on_osx || return 1
  fi
  kube::build::ensure_docker_daemon_connectivity || return 1

  KUBE_ROOT_HASH=$(kube::build::short_hash "${HOSTNAME:-}:${REPO_DIR:-${KUBE_ROOT}}/go-to-protobuf")
  KUBE_BUILD_IMAGE_TAG="build-${KUBE_ROOT_HASH}"
  KUBE_BUILD_IMAGE="${KUBE_BUILD_IMAGE_REPO}:${KUBE_BUILD_IMAGE_TAG}"
  KUBE_BUILD_CONTAINER_NAME="kube-build-${KUBE_ROOT_HASH}"
  KUBE_BUILD_DATA_CONTAINER_NAME="kube-build-data-${KUBE_ROOT_HASH}"
  DOCKER_MOUNT_ARGS=(
    --volume "${REPO_DIR:-${KUBE_ROOT}}:/go/src/${KUBE_GO_PACKAGE}"
    --volume /etc/localtime:/etc/localtime:ro
    --volumes-from "${KUBE_BUILD_DATA_CONTAINER_NAME}"
  )
  LOCAL_OUTPUT_BUILD_CONTEXT="${LOCAL_OUTPUT_IMAGE_STAGING}/${KUBE_BUILD_IMAGE}"
}

prereqs
mkdir -p "${LOCAL_OUTPUT_BUILD_CONTEXT}"
cp "${KUBE_ROOT}/cmd/libs/go2idl/go-to-protobuf/build-image/Dockerfile" "${LOCAL_OUTPUT_BUILD_CONTEXT}/Dockerfile"
kube::build::update_dockerfile
kube::build::docker_build "${KUBE_BUILD_IMAGE}" "${LOCAL_OUTPUT_BUILD_CONTEXT}" 'false'
kube::build::run_build_command hack/update-generated-runtime-dockerized.sh "$@"

