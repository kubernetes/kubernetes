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

# This script checks whether the latest or untagged  gcr.io image is in
# `test/e2e/*.go` files.
# Usage: `hack/verify-test-images.sh`.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..

cd "${KUBE_ROOT}"
result=0

# Find mentions of untagged gcr.io images in test/e2e/*.go
find_e2e_test_untagged_gcr_images() {
    grep -o -E -e 'gcr.io/[-a-z0-9/_:.]+' test/e2e/*.go | grep -v -E "gcr.io/.*:" | cut -d ":" -f 1 | LC_ALL=C sort -u
}


# Find mentions of latest gcr.io images in test/e2e/*.go
find_e2e_test_latest_gcr_images() {
    grep -o -E -e 'gcr.io/.*:latest' test/e2e/*.go | cut -d ":" -f 1 | LC_ALL=C sort -u
}

if find_e2e_test_latest_gcr_images; then
  echo "!!! Found :latest gcr.io images in the above files"
  result=1
fi

if find_e2e_test_untagged_gcr_images; then
  echo "!!! Found untagged gcr.io images in the above files"
  result=1
fi

exit ${result}
