#!/bin/bash

# Copyright 2015 The Kubernetes Authors.
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

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..

: ${KUBECTL:=${KUBE_ROOT}/cluster/kubectl.sh}
: ${KUBE_CONFIG_FILE:="config-test.sh"}

export KUBECTL KUBE_CONFIG_FILE

source "${KUBE_ROOT}/cluster/kube-util.sh"

prepare-e2e

if [[ "${FEDERATION:-}" == "true" ]];then
  source "${KUBE_ROOT}/federation/cluster/common.sh"
  for zone in ${E2E_ZONES};do
    # bring down an e2e cluster
    (
      set-federation-zone-vars "$zone"
      cleanup-federation-api-objects || echo "Couldn't cleanup federation api objects"

      # TODO(madhusudancs): This is an arbitrary amount of sleep to give Kubernetes
      # clusters enough time to delete the underlying cloud provider resources
      # corresponding to the Kubernetes resources we deleted as part of the test
      # teardowns. It is shameful that we are doing this, but this is just a bandage
      # to stop the bleeding. Please don't use this pattern anywhere. Remove this
      # when proper cloud provider cleanups are implemented in the individual test
      # `AfterEach` blocks.
      sleep 2m

      test-teardown
    )
done
else
  test-teardown
fi
