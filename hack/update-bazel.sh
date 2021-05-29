#!/usr/bin/env bash
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

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..

# delete all bazel related files not in vendor/
find "${KUBE_ROOT}" \
  \( \
    -name BUILD \
    -o -name BUILD.bazel \
    -o -name '*.bzl' \
  \) \
  -not \
  \( \
    -path "${KUBE_ROOT}"'/vendor*' \
  \) \
  -delete

# remove additional one-off bazel related files
# NOTE: most of these will be pairs of symlinked location in "${KUBE_ROOT}/"
# and the actual location in "${KUBE_ROOT}/build/root/"
rm -f \
  "${KUBE_ROOT}/build/root/BUILD.root" \
  "${KUBE_ROOT}/WORKSPACE" \
  "${KUBE_ROOT}/build/root/WORKSPACE" \
  "${KUBE_ROOT}/.bazelrc" \
  "${KUBE_ROOT}/build/root/.bazelrc" \
  "${KUBE_ROOT}/.bazelversion" \
  "${KUBE_ROOT}/build/root/.bazelversion" \
  "${KUBE_ROOT}/.kazelcfg.json" \
  "${KUBE_ROOT}/build/root/.kazelcfg.json"

