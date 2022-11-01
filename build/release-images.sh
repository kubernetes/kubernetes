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
if [[ "${KUBE_BUILD_CONFORMANCE}" =~ [yY] ]]; then
    CMD_TARGETS="${CMD_TARGETS} ${KUBE_CONFORMANCE_IMAGE_TARGETS[*]}"
fi
# include extra WHAT if specified so you can build docker images + binaries
# in one call with a single pass of syncing to the container + generating code
if [[ -n "${KUBE_EXTRA_WHAT:-}" ]]; then
    CMD_TARGETS="${CMD_TARGETS} ${KUBE_EXTRA_WHAT}"
fi

kube::build::verify_prereqs
kube::build::build_image
kube::build::run_build_command make all WHAT="${CMD_TARGETS}" KUBE_BUILD_PLATFORMS="${KUBE_SERVER_PLATFORMS[*]}" DBG="${DBG:-}"

kube::build::copy_output

kube::release::build_server_images
