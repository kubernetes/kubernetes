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

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::log::status "Restoring kubernetes deps"

if [ -z "${SKIP_KDEP_INSTALL:-}" ]; then
  kube::util::ensure_kdep_version
fi

kube::log::status "Downloading dependencies - this might take a while"
kdep ensure "$@"
kube::log::status "Done"

kube::log::status "Refreshing bazel build files"
"${KUBE_ROOT}/hack/update-bazel.sh"
kube::log::status "Done"
