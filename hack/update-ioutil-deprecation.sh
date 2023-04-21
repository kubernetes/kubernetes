#!/usr/bin/env bash

# Copyright 2023 The Kubernetes Authors.
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

#This script update the source code for deprecated methods in the
# ioutil package that need to be replaced

# Usage: `hack/update-ioutil-deprecation.sh`.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

cd "${KUBE_ROOT}"

find_and_replace_files() {
  find . -not \( \
      \( \
        -wholename './output' \
        -o -wholename './.git' \
        -o -wholename './_output' \
        -o -wholename './_gopath' \
        -o -wholename './release' \
        -o -wholename './target' \
        -o -wholename '*/vendor/*' \
      \) -prune \
    \) -name '*.go' -execdir bash -c 'replace_ioutil "$@"' _ {} \; 2>&1
}

replace_ioutil() {
  local file="$1"

  sed -i 's/ioutil.ReadFile/os.ReadFile/g' "$file"

  sed -i 's/ioutil.WriteFile/os.WriteFile/g' "$file"

  sed -i 's/ioutil.Discard/io.Discard/g' "$file"

  sed -i 's/ioutil.NopCloser/io.NopCloser/g' "$file"

  sed -i 's/ioutil.TempFile/os.CreateTemp/g' "$file"

  sed -i 's/ioutil.ReadAll/io.ReadAll/g' "$file"
}

export -f replace_ioutil

find_and_replace_files 2>&1