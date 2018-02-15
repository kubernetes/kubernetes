#!/usr/bin/env bash

# Copyright 2017 The Kubernetes Authors.
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

# Verify whether files containing build tags are named with a platform specific
# suffix. This is needed to ensure that the cross build is run when a file with
# build tags changed.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

# Excluded files and directories
EXCLUDED_FILES=(
  "vendor/*"
  "**/zz_generated*.go"
)

# File extensions that are permitted to have build tags
SUPPORTED_EXTENSIONS=(
  "linux"
  "osx"
  "solaris"
  "unix"
  "unsupported"
  "windows"
)

FILTER_PATTERNS=()
for i in "${EXCLUDED_FILES[@]}"; do
    FILTER_PATTERNS+=(":(exclude)${i}")
    FILTER_PATTERNS+=(":(exclude)${i}")
done
for i in "${SUPPORTED_EXTENSIONS[@]}"; do
    FILTER_PATTERNS+=(":(exclude)**/*_${i}.go")
    FILTER_PATTERNS+=(":(exclude)**/*_${i}_test.go")
done

cd "${KUBE_ROOT}"
if git --no-pager grep -E $'^(// \+build) .*$' -- '**/*.go' "${FILTER_PATTERNS[@]}"; then
  kube::log::error "Build tags are present in files that don't match a support build extension"
  exit 1
fi
