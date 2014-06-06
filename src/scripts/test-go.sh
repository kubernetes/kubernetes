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


set -e

source $(dirname $0)/config-go.sh


find_test_dirs() {
  (
    cd src/${KUBE_GO_PACKAGE}
    find . -not \( \
        \( \
          -wholename './third_party' \
          -o -wholename './release' \
          -o -wholename './target' \
        \) -prune \
      \) -name '*_test.go' -print0 | xargs -0n1 dirname | sort -u
  )
}


cd "${KUBE_TARGET}"
for package in $(find_test_dirs); do
  go test -cover -coverprofile="tmp.out" "${KUBE_GO_PACKAGE}/${package}"
done
