#!/usr/bin/env bash

# Copyright The Kubernetes Authors.
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

# This script updates the tools references in hack/tools
# apart from tools with specific subdirectories (golangci-lint,
# instrumentation).
# Each tool that has an available upstream update is bumped in a
# separate commit.
# Usage: `hack/update-tools.sh`.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::golang::setup_env

if ! git -C "${KUBE_ROOT}" diff --quiet hack/tools; then
    echo Please start run "${0}" with a clean hack/tools tree.
    exit 1
fi

for tool in $(go -C "${KUBE_ROOT}/hack/tools" tool | grep /); do
    echo Checking "${tool}"...
    go -C "${KUBE_ROOT}/hack/tools" get "${tool}"
    go -C "${KUBE_ROOT}/hack/tools" mod tidy
    git -C "${KUBE_ROOT}" diff --quiet hack/tools || git -C "${KUBE_ROOT}" commit -s -m "Bump ${tool}" hack/tools
done
