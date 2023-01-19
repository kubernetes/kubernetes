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

# This script generates `*/api.pb.go` from the protobuf file `*/api.proto`.
# Example:
#   kube::protoc::generate_proto "${KUBELET_PLUGIN_REGISTRATION_V1ALPHA}"

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd -P)"

dirs=(
    "staging/src/k8s.io/kubelet/pkg/apis/pluginregistration/v1alpha1/"
    "staging/src/k8s.io/kubelet/pkg/apis/pluginregistration/v1beta1/"
    "staging/src/k8s.io/kubelet/pkg/apis/pluginregistration/v1/"
    "pkg/kubelet/pluginmanager/pluginwatcher/example_plugin_apis/v1beta1/"
    "pkg/kubelet/pluginmanager/pluginwatcher/example_plugin_apis/v1beta2/"
)

source "${KUBE_ROOT}/hack/lib/protoc.sh"

for d in "${dirs[@]}"; do
    kube::protoc::generate_proto "${KUBE_ROOT}/${d}"
done
