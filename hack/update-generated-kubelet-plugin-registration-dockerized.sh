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
KUBELET_PLUGIN_REGISTRATION_ROOT="${KUBE_ROOT}/pkg/kubelet/apis/pluginregistration"
KUBELET_PLUGIN_REGISTRATION_VERSION="$(find "${KUBELET_PLUGIN_REGISTRATION_ROOT}" -maxdepth 1 -type d -path "${KUBELET_PLUGIN_REGISTRATION_ROOT}/*" -printf "%f\n")"

KUBELET_EXAMPLE_PLUGIN_ROOT="${KUBE_ROOT}/pkg/kubelet/pluginmanager/pluginwatcher/example_plugin_apis"
KUBELET_EXAMPLE_PLUGIN_VERSION="$(find "${KUBELET_EXAMPLE_PLUGIN_ROOT}" -maxdepth 1 -type d -path "${KUBELET_EXAMPLE_PLUGIN_ROOT}/*" -printf "%f\n")"

source "${KUBE_ROOT}/hack/lib/protoc.sh"
# generate plugin registration proto
for VERSION in ${KUBELET_PLUGIN_REGISTRATION_VERSION}
do
    kube::protoc::generate_proto "${KUBELET_PLUGIN_REGISTRATION_ROOT}/${VERSION}"
    echo "Generated kubelet plugin registration ${VERSION} api is up to date."
done

# generate example plugin proto
for VERSION in ${KUBELET_EXAMPLE_PLUGIN_VERSION}
do
    kube::protoc::generate_proto "${KUBELET_EXAMPLE_PLUGIN_ROOT}/${VERSION}"
    echo "Generated kubelet example plugin ${VERSION} api is up to date."
done
