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

# This script checks import restrictions. The script looks for a file called
# `.import-restrictions` in each directory, then all imports of the package are
# checked against each "rule" in the file.
# Usage: `hack/verify-import-boss.sh`.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::golang::setup_env
kube::util::require-jq

# Doing it this way is MUCH faster than simply saying "all", and there doesn't
# seem to be a simpler way to express "this whole workspace".
packages=()
kube::util::read-array packages < <(
    go work edit -json | jq -r '.Use[].DiskPath + "/..."'
)

GOPROXY=off \
    go run ./cmd/import-boss -v "${KUBE_VERBOSE:-0}" "${packages[@]}"
