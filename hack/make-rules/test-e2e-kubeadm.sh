#!/usr/bin/env bash

# Copyright 2019 The Kubernetes Authors.
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

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/../..
source "${KUBE_ROOT}/hack/lib/init.sh"

focus=${FOCUS:-""}
skip=${SKIP:-""}
parallelism=${PARALLELISM:-""}
artifacts=${ARTIFACTS:-"/tmp/_artifacts/$(date +%y%m%dT%H%M%S)"}
run_until_failure=${RUN_UNTIL_FAILURE:-"false"}
build=${BUILD:-"true"}

# Parse the flags to pass to ginkgo
ginkgoflags=""
if [[ ${parallelism} != "" ]]; then
  ginkgoflags="${ginkgoflags} -nodes=${parallelism} "
else
  ginkgoflags="${ginkgoflags} -p "
fi

if [[ ${focus} != "" ]]; then
  ginkgoflags="${ginkgoflags} -focus=\"${focus}\" "
fi

if [[ ${skip} != "" ]]; then
  ginkgoflags="${ginkgoflags} -skip=\"${skip}\" "
fi

if [[ ${run_until_failure} != "false" ]]; then
  ginkgoflags="${ginkgoflags} -untilItFails=${run_until_failure} "
fi

# Setup the directory to copy test artifacts (logs, junit.xml, etc) from remote host to local host
if [[ ! -d "${artifacts}" ]]; then
  echo "Creating artifacts directory at ${artifacts}"
  mkdir -p "${artifacts}"
fi
echo "Test artifacts will be written to ${artifacts}"

# Test 
kube::golang::verify_go_version

go run test/e2e_kubeadm/runner/local/run_local.go \
  --ginkgo-flags="${ginkgoflags}" \
  --test-flags="--provider=skeleton --report-dir=${artifacts}" \
  --build="${build}" 2>&1 | tee -i "${artifacts}/build-log.txt"
