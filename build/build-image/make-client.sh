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

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
source "${KUBE_ROOT}/build/build-image/common.sh"

platforms=(linux/amd64 $KUBE_CROSSPLATFORMS)
targets=("${client_targets[@]}")

if [[ $# -gt 0 ]]; then
  targets=("$@")
fi

for platform in "${platforms[@]}"; do
  (
    # Subshell to contain these exports
    export GOOS=${platform%/*}
    export GOARCH=${platform##*/}

    kube::build::make_binaries "${targets[@]}"
  )
done
