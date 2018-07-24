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

# Run a command in the docker build container.  Typically this will be one of
# the commands in `hack/`.  When running in the build container the user is sure
# to have a consistent reproducible build environment.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "$KUBE_ROOT/build/common.sh"

KUBE_RUN_COPY_OUTPUT="${KUBE_RUN_COPY_OUTPUT:-y}"

kube::build::verify_prereqs
kube::build::build_image

if [[ ${KUBE_RUN_COPY_OUTPUT} =~ ^[yY]$ ]]; then
  kube::log::status "Output from this container will be rsynced out upon completion. Set KUBE_RUN_COPY_OUTPUT=n to disable."
else
  kube::log::status "Output from this container will NOT be rsynced out upon completion. Set KUBE_RUN_COPY_OUTPUT=y to enable."
fi

kube::build::run_build_command "$@"

if [[ ${KUBE_RUN_COPY_OUTPUT} =~ ^[yY]$ ]]; then
  kube::build::copy_output
fi
