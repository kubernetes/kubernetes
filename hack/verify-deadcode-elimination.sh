#!/usr/bin/env bash

# Copyright 2025 The Kubernetes Authors.
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

# This script verifies if the golang linker is eliminating dead code in
# various components we care about, such as kube-apiserver, kubelet and others
# Usage: `hack/verify-deadcode-elimination.sh`.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::golang::setup_env

# Ensure that we find the binaries we build before anything else.
export GOBIN="${KUBE_OUTPUT_BIN}"
PATH="${GOBIN}:${PATH}"

# Install whydeadcode
go install github.com/aarzilli/whydeadcode@latest

# Prefer full path for running zeitgeist
WHYDEADCODE_BIN="$(which whydeadcode)"

pushd "${KUBE_ROOT}"

# Define an array of binaries to check
BINARIES=("kube-apiserver" "kubelet" "kube-controller-manager" "kube-scheduler" "kube-proxy")
FAILED=false
FAILED_BINARIES=()

for binary in "${BINARIES[@]}"; do
  echo "Processing ${binary} ..."
  output=$(KUBE_VERBOSE=4 GOLDFLAGS=-dumpdep make "${binary}" 2>&1 | grep "\->" | ${WHYDEADCODE_BIN} 2>&1)
  if [[ -n "$output" ]]; then
    echo "golang linker is not eliminating dead code in ${binary}, please check the trace output below:"
    echo "(NOTE: that there may be false positives, but the first trace should be a real issue)"
    echo "$output"
    FAILED=true
    FAILED_BINARIES+=("${binary}")
  fi
  
  # Find the binary and print its size
  echo "Finding ${binary} binary and checking its size:"
  binary_paths=$(find _output -type f -name "${binary}" | sort)
  if [[ -n "${binary_paths}" ]]; then
    # shellcheck disable=SC2086
    ls -altrh ${binary_paths}
  else
    echo "Binary ${binary} not found in _output directory"
  fi
  echo ""
done

popd > /dev/null || true

if [[ "$FAILED" == "true" ]]; then
  echo "Dead code elimination check failed for the following binaries:"
  for failed_binary in "${FAILED_BINARIES[@]}"; do
    echo "  - ${failed_binary}"
  done
  exit 1
fi
