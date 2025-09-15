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

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/../../../../..
source "${KUBE_ROOT}/hack/lib/util.sh"

# Register function to be called on EXIT to remove generated binary.
function cleanup {
  rm "${KUBE_ROOT}/staging/src/k8s.io/kube-aggregator/artifacts/simple-image/kube-aggregator"
}
trap cleanup EXIT

pushd "${KUBE_ROOT}/staging/src/k8s.io/kube-aggregator"
cp -v ../../../../_output/local/bin/linux/amd64/kube-aggregator ./artifacts/simple-image/kube-aggregator
docker build -t kube-aggregator:latest ./artifacts/simple-image
popd
