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

# Cleans output files/images and builds a full release from scratch
#
# Prerequisite:
# ./cluster/mesos/docker/test/build.sh
#
# Example Usage:
# ./contrib/mesos/ci/build-release.sh

set -o errexit
set -o nounset
set -o pipefail
set -o errtrace

KUBE_ROOT=$(cd "$(dirname "${BASH_SOURCE}")/../../.." && pwd)

export KUBERNETES_CONTRIB="${KUBERNETES_CONTRIB:-mesos}"
export KUBE_RELEASE_RUN_TESTS="${KUBE_RELEASE_RUN_TESTS:-N}"
export KUBE_SKIP_CONFIRMATIONS=Y

## "${KUBE_ROOT}/build/release.sh" -- hacked to call our own build-cross
source "$KUBE_ROOT/build/common.sh"

KUBE_RELEASE_RUN_TESTS=${KUBE_RELEASE_RUN_TESTS-y}

kube::build::verify_prereqs
kube::build::build_image
kube::build::run_build_command contrib/mesos/ci/build-cross.sh

if [[ $KUBE_RELEASE_RUN_TESTS =~ ^[yY]$ ]]; then
  kube::build::run_build_command hack/test-go.sh
  kube::build::run_build_command hack/test-integration.sh
fi

# TODO:
#kube::build::copy_output
#kube::release::package_tarballs
