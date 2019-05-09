#!/usr/bin/env bash

# Copyright 2017 The Kubernetes Authors.
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
DEVICE_PLUGIN_ALPHA="${KUBE_ROOT}/pkg/kubelet/apis/deviceplugin/v1alpha/"
DEVICE_PLUGIN_V1BETA1="${KUBE_ROOT}/pkg/kubelet/apis/deviceplugin/v1beta1/"

source "${KUBE_ROOT}/hack/lib/protoc.sh"
kube::protoc::generate_proto "${DEVICE_PLUGIN_ALPHA}"
kube::protoc::generate_proto "${DEVICE_PLUGIN_V1BETA1}"
