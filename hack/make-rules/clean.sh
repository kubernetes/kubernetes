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

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
source "${KUBE_ROOT}/hack/lib/util.sh"

CLEAN_PATTERNS=(
  "_tmp"
  "doc_tmp"
  ".*/zz_generated.openapi.go"
  "test/e2e/generated/bindata.go"
)

for pattern in ${CLEAN_PATTERNS[@]}; do
  for match in $(find "${KUBE_ROOT}" -iregex "^${KUBE_ROOT}/${pattern}$"); do
    echo "Removing ${match#${KUBE_ROOT}\/} .."
    rm -rf "${match#${KUBE_ROOT}\/}"
  done
done

# ex: ts=2 sw=2 et filetype=sh
