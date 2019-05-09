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

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
ERROR="Kubelet Plugin Registration api is out of date. Please run hack/update-generated-kubelet-plugin-registration.sh"
KUBELET_PLUGIN_REGISTRATION_V1ALPHA="${KUBE_ROOT}/pkg/kubelet/apis/pluginregistration/v1alpha1/"
KUBELET_PLUGIN_REGISTRATION_V1BETA="${KUBE_ROOT}/pkg/kubelet/apis/pluginregistration/v1beta1/"

source "${KUBE_ROOT}/hack/lib/protoc.sh"
kube::golang::setup_env

function cleanup {
	rm -rf "${KUBELET_PLUGIN_REGISTRATION_V1ALPHA}/_tmp/"
	rm -rf "${KUBELET_PLUGIN_REGISTRATION_V1BETA}/_tmp/"
}

trap cleanup EXIT

mkdir -p "${KUBELET_PLUGIN_REGISTRATION_V1ALPHA}/_tmp"
mkdir -p "${KUBELET_PLUGIN_REGISTRATION_V1BETA}/_tmp"

cp "${KUBELET_PLUGIN_REGISTRATION_V1ALPHA}/api.pb.go" "${KUBELET_PLUGIN_REGISTRATION_V1ALPHA}/_tmp/"
cp "${KUBELET_PLUGIN_REGISTRATION_V1BETA}/api.pb.go" "${KUBELET_PLUGIN_REGISTRATION_V1BETA}/_tmp/"

# Check V1Alpha
KUBE_VERBOSE=3 "${KUBE_ROOT}/hack/update-generated-kubelet-plugin-registration.sh"
kube::protoc::diff "${KUBELET_PLUGIN_REGISTRATION_V1ALPHA}/api.pb.go" "${KUBELET_PLUGIN_REGISTRATION_V1ALPHA}/_tmp/api.pb.go" "${ERROR}"
echo "Generated Kubelet Plugin Registration api is up to date."

# Check V1Beta
KUBE_VERBOSE=3 "${KUBE_ROOT}/hack/update-generated-kubelet-plugin-registration.sh"
kube::protoc::diff "${KUBELET_PLUGIN_REGISTRATION_V1BETA}/api.pb.go" "${KUBELET_PLUGIN_REGISTRATION_V1BETA}/_tmp/api.pb.go" "${ERROR}"
echo "Generated Kubelet Plugin Registration api is up to date."
