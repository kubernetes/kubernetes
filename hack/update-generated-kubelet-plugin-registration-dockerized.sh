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
KUBELET_PLUGIN_REGISTRATION_V1ALPHA="${KUBE_ROOT}/pkg/kubelet/apis/pluginregistration/v1alpha1/"
KUBELET_PLUGIN_REGISTRATION_V1BETA="${KUBE_ROOT}/pkg/kubelet/apis/pluginregistration/v1beta1/"
KUBELET_EXAMPLE_PLUGIN_V1BETA1="${KUBE_ROOT}/pkg/kubelet/pluginmanager/pluginwatcher/example_plugin_apis/v1beta1/"
KUBELET_EXAMPLE_PLUGIN_V1BETA2="${KUBE_ROOT}/pkg/kubelet/pluginmanager/pluginwatcher/example_plugin_apis/v1beta2/"

source "${KUBE_ROOT}/hack/lib/protoc.sh"
kube::protoc::generate_proto "${KUBELET_PLUGIN_REGISTRATION_V1ALPHA}"
kube::protoc::generate_proto "${KUBELET_PLUGIN_REGISTRATION_V1BETA}"
kube::protoc::generate_proto "${KUBELET_EXAMPLE_PLUGIN_V1BETA1}"
kube::protoc::generate_proto "${KUBELET_EXAMPLE_PLUGIN_V1BETA2}"
