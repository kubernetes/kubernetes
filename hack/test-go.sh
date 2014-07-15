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
          -o -wholename '*/third_party/*' \
          -o -wholename '*/output/*' \
        \) -prune \
      \) -name '*_test.go' -print0 | xargs -0n1 dirname | sort -u
  )
}

# -covermode=atomic becomes default with -race in Go >=1.3
KUBE_COVER="-cover -covermode=atomic -coverprofile=\"tmp.out\""

cd "${KUBE_TARGET}"

if [ "$1" != "" ]; then
  go test -race $KUBE_COVER "$KUBE_GO_PACKAGE/$1" "${@:2}"
  exit 0
fi

for package in $(find_test_dirs); do
  go test -race $KUBE_COVER "${KUBE_GO_PACKAGE}/${package}" "${@:2}"
done
