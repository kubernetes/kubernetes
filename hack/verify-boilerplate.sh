#!/bin/bash

# Copyright 2014 The Kubernetes Authors All rights reserved.
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

cd ${KUBE_ROOT}

result=0
find_files() {
  find . -not \( \
      \( \
        -wholename './output' \
        -o -wholename './_output' \
        -o -wholename './release' \
        -o -wholename './target' \
        -o -wholename '*/third_party/*' \
        -o -wholename '*/Godeps/*' \
      \) -prune \
    \) -name '*.go'
}

for file in $(find_files); do
  if [[ "$("${KUBE_ROOT}/hooks/boilerplate.sh" "${file}")" -eq "0" ]]; then
    echo "Boilerplate header is wrong for: ${file}"
    result=1
  fi
done

dirs=("cluster" "hack" "hooks" "build")

for dir in ${dirs[@]}; do
  for file in $(find "$dir" -name '*.sh'); do
    if [[ "$("${KUBE_ROOT}/hooks/boilerplate.sh" "${file}")" -eq "0" ]]; then
      echo "Boilerplate header is wrong for: ${file}"
      result=1
    fi
  done
done


exit ${result}
