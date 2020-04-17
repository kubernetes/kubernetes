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

# This script sets up a go workspace locally and builds all for all appropriate
# platforms.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/../..
source "${KUBE_ROOT}/hack/lib/init.sh"

# NOTE: Using "${array[*]}" here is correct.  [@] becomes distinct words (in
# bash parlance).

make all WHAT="${KUBE_SERVER_TARGETS[*]}" KUBE_BUILD_PLATFORMS="${KUBE_SERVER_PLATFORMS[*]}"

make all WHAT="${KUBE_NODE_TARGETS[*]}" KUBE_BUILD_PLATFORMS="${KUBE_NODE_PLATFORMS[*]}"

make all WHAT="${KUBE_CLIENT_TARGETS[*]}" KUBE_BUILD_PLATFORMS="${KUBE_CLIENT_PLATFORMS[*]}"

make all WHAT="${KUBE_TEST_TARGETS[*]}" KUBE_BUILD_PLATFORMS="${KUBE_TEST_PLATFORMS[*]}"

make all WHAT="${KUBE_TEST_SERVER_TARGETS[*]}" KUBE_BUILD_PLATFORMS="${KUBE_TEST_SERVER_PLATFORMS[*]}"
