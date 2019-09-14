#!/usr/bin/env bash

# Copyright 2018 The Kubernetes Authors.
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

KUBE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd -P)"
POD_RESOURCES_ROOT="${KUBE_ROOT}/pkg/kubelet/apis/podresources"
POD_RESOURCES_VERSION=$(find "${POD_RESOURCES_ROOT}" -maxdepth 1 -type d -path "${POD_RESOURCES_ROOT}/*" -printf "%f\n")

source "${KUBE_ROOT}/hack/lib/protoc.sh"
# generate pod resource proto
for VERSION in ${POD_RESOURCES_VERSION}
do
    kube::protoc::generate_proto "${POD_RESOURCES_ROOT}/${VERSION}"
    echo "Generated pod resources ${VERSION} api is up to date."
done
