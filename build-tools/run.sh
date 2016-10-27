#!/bin/bash

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
source "$KUBE_ROOT/build-tools/common.sh"

kube::build::verify_prereqs
kube::build::build_image
kube::build::run_build_command "$@"

if [[ ${KUBE_RUN_COPY_OUTPUT:-y} =~ ^[yY]$ ]]; then
  kube::build::copy_output
fi
