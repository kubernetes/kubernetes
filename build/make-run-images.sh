#!/bin/bash

# Copyright 2014 Google Inc. All rights reserved.
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

# Build the docker image necessary for running Kubernetes
#
# This script will make the 'run image' after building all of the necessary
# binaries.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
KUBE_BUILD_RUN_IMAGES=y
source "$KUBE_ROOT/build/common.sh"

kube::build::verify_prereqs
kube::build::build_image
kube::build::run_build_command hack/build-go.sh "$@"
kube::build::copy_output
kube::build::run_image
