#!/bin/bash

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

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..

source "${KUBE_ROOT}/cluster/kube-util.sh"

#A little hack to get the last zone. we always deploy federated cluster to the last zone.
#TODO(colhom): deploy federated control plane to multiple underlying clusters in robust way
lastZone=""
for zone in ${E2E_ZONES};do
    lastZone="$zone"
done
(
    set-federation-zone-vars "$zone"
    "${KUBE_ROOT}/hack/ginkgo-e2e.sh" $@
)
