#!/usr/bin/env bash

# Copyright 2018 The Kubernetes Authors.
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

# Build Kubernetes release images. This will build the server target binaries,
# and create wrap them in Docker images, see `make release` for full releases

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/build/common.sh"
source "${KUBE_ROOT}/build/lib/release.sh"

CMD_TARGETS="${KUBE_SERVER_IMAGE_TARGETS[*]}"
if [[ "${KUBE_BUILD_HYPERKUBE}" =~ [yY] ]]; then
    CMD_TARGETS="${CMD_TARGETS} cmd/hyperkube"
fi
if [[ "${KUBE_BUILD_CONFORMANCE}" =~ [yY] ]]; then
    CMD_TARGETS="${CMD_TARGETS} ${KUBE_CONFORMANCE_IMAGE_TARGETS[*]}"
fi

kube::build::verify_prereqs
kube::build::build_image
kube::build::run_build_command make all WHAT="${CMD_TARGETS}" KUBE_BUILD_PLATFORMS="${KUBE_SERVER_PLATFORMS[*]}"

kube::build::copy_output

kube::release::build_server_images
