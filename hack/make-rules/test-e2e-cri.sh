#!/bin/bash

# Copyright 2016 The Kubernetes Authors.
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

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
source "${KUBE_ROOT}/hack/lib/init.sh"

focus=${FOCUS:-""}
skip=${SKIP:-""}
report=${REPORT:-"/tmp/"}
run_until_failure=${RUN_UNTIL_FAILURE:-"false"}
test_args=${TEST_ARGS:-""}
runtime_service_address=${CONTAINER_RUNTIME_ENDPOINT:-"/var/run/dockershim.sock"}
image_service_address=${IMAGE_SERVICE_ENDPOINT:-""}

# Parse the flags to pass to ginkgo
ginkgoflags=""
if [[ $focus != "" ]]; then
  ginkgoflags="$ginkgoflags -focus=$focus "
fi

if [[ $skip != "" ]]; then
  ginkgoflags="$ginkgoflags -skip=$skip "
fi

if [[ $run_until_failure != "" ]]; then
  ginkgoflags="$ginkgoflags -untilItFails=$run_until_failure "
fi

# Test using the host the script was run on
# Provided for backwards compatibility
go run test/e2e_cri/runner/run.go --ginkgo-flags="$ginkgoflags" \
  --test-flags="--alsologtostderr --v 4 --report-dir=${report} \
  $test_args" --runtime-service-address=${runtime_service_address} \
  --image-service-address=${image_service_address} --build-dependencies=true
exit $?
