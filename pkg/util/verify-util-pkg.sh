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

# verify-util-pkg.sh checks whether *.go except doc.go in pkg/util have been moved into
# sub-pkgs, see issue #15634.

set -o errexit
set -o nounset
set -o pipefail

BASH_DIR=$(dirname "${BASH_SOURCE[0]}")

find_go_files() {
  find . -maxdepth 1 -not \( \
      \( \
        -wholename './doc.go' \
      \) -prune \
    \) -name '*.go'
}

ret=0

pushd "${BASH_DIR}" > /dev/null
  for path in $(find_go_files); do
    file=$(basename "$path")
    echo "Found pkg/util/${file}, but should be moved into util sub-pkgs." 1>&2
    ret=1
  done
popd > /dev/null

if [[ ${ret} -gt 0 ]]; then
  exit ${ret}
fi

echo "Util Package Verified."
