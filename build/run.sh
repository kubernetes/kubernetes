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

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "$KUBE_ROOT/build/common.sh"

# This is no longer supported, warn if set to non-default value
KUBE_RUN_COPY_OUTPUT="${KUBE_RUN_COPY_OUTPUT:-y}"
# we previously accepted an explicit both y and Y
if [[ ! "${KUBE_RUN_COPY_OUTPUT}" =~ ^[yY]$ ]]; then
   kube::log::error "KUBE_RUN_COPY_OUTPUT no longer means anything as we bind-mount instead of rsyncing, so output is always persisted"
fi

kube::build::verify_prereqs
kube::build::run_build_command "$@"
