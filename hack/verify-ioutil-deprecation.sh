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

#This script checks the source code for deprecated methods in the
# ioutil package that need to be replaced

# Usage: `hack/verify-ioutil-deprecation.sh`.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

cd "${KUBE_ROOT}"

error_messages=""

find_files() {
  # shellcheck disable=SC2207
  files=($(find . -not \( \
      \( \
        -wholename './output' \
        -o -wholename './.git' \
        -o -wholename './_output' \
        -o -wholename './_gopath' \
        -o -wholename './release' \
        -o -wholename './target' \
        -o -wholename '*/vendor/*' \
      \) -prune \
    \) -name '*.go'))
}

find_files

for file in "${files[@]}"; do
  if grep -q "ioutil.ReadFile" "$file"; then
    error_messages+="ERROR: $file: ioutil.ReadFile usage found\n"
  fi

  if grep -q "ioutil.WriteFile" "$file"; then
    error_messages+="ERROR: $file: ioutil.WriteFile usage found\n"
  fi

  if grep -q "ioutil.Discard" "$file"; then
    error_messages+="ERROR: $file: ioutil.Discard usage found\n"
  fi

  if grep -q "ioutil.NopCloser" "$file"; then
    error_messages+="ERROR: $file: ioutil.NopCloser usage found\n"
  fi

  if grep -q "ioutil.TempFile" "$file"; then
    error_messages+="ERROR: $file: ioutil.TempFile usage found\n"
  fi

  if grep -q "ioutil.ReadAll" "$file"; then
    error_messages+="ERROR: $file: ioutil.ReadAll usage found\n"
  fi
done

if [[ -n $error_messages ]]; then
  echo >&2 -e "$error_messages"
  echo >&2 "Run ./hack/update-ioutil-deprecation.sh"
  exit 1
fi