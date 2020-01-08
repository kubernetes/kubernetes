#!/usr/bin/env bash

# Copyright 2020 The Kubernetes Authors.
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

# GoFmt apparently is changing @ head...

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

cd "${KUBE_ROOT}"

find_files() {
  # TODO: Need to check other go files also
  find test/e2e/storage/ -name '*.go'
}

invalid_files=()
for gofile in `find_files`
do
	if [ -n "$(file -i ${gofile} | grep utf-8)" ]
	then
		invalid_files+=( "${gofile}" )
	fi
done

if [ ${#invalid_files[@]} -ne 0 ]; then
  {
    echo "Errors:"
    for err in "${invalid_files[@]}"; do
      echo "$err"
    done
    echo
    echo 'The above files contains non-ascii string, need to remove it'
    echo
  } >&2
  exit 1
fi
