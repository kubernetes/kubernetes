#!/usr/bin/env bash

# Copyright 2021 The Kubernetes Authors.
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

# This script runs to ensure that we do not violate metric stability
# policies.
# Usage: `hack/verify-generated-stable-metrics.sh`.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"
source "${KUBE_ROOT}/hack/lib/util.sh"

kube::golang::verify_go_version
extra_gopath=$(mktemp -d)
cd "${KUBE_ROOT}"
export KUBE_EXTRA_GOPATH=$extra_gopath
kube::golang::setup_env

find_files() {
  find . -not \( \
      \( \
        -wholename './output' \
        -o -wholename './_output' \
        -o -wholename './_gopath' \
        -o -wholename './release' \
        -o -wholename './target' \
        -o -wholename '*/third_party/*' \
        -o -wholename '*/vendor/*' \
        -o -wholename '*/hack/*' \
        -o -wholename '**/*_test.go' \
        \) -prune \
    \) \
    \( -wholename '**/*.go' \
  \)
}

static_checked_files=$(find_files | grep -E ".*.go" | grep -v ".*_test.go") || true
pushd "${KUBE_EXTRA_GOPATH}" >/dev/null
  GO111MODULE=on go get "gopkg.in/yaml.v2"
popd >/dev/null

for i in "${static_checked_files[@]}"
do
  temp_file=$(mktemp)
  gopathfiles=$(find_files | grep -E 'test/instrumentation/.*.go' | grep -v "test/instrumentation/main.*go" | cut -c 3-)
  # Deliberately allow word split here
  # shellcheck disable=SC2086
  stabilitycheck=$(go run "test/instrumentation/main.go" $gopathfiles -- $i 1>$temp_file)
  if ! $stabilitycheck; then
    echo "!!! Metrics Stability static analysis has failed!" >&2
    echo "!!! Please run ./hack/update-generated-stable-metrics.sh to update the golden list." >&2
    exit 1
  fi
  if diff -u "$KUBE_ROOT/test/instrumentation/testdata/stable-metrics-list.yaml" "$temp_file"; then
    echo "PASS metrics stability verification"
  else
    echo "!!! Metrics Stability static analysis has failed!" >&2
    echo "!!! Please run ./hack/update-generated-stable-metrics.sh to update the golden list." >&2
    echo "!!! $temp_file" >&2
    exit 1
  fi
done


