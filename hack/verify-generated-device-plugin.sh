#!/bin/bash

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

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
ERROR="Device plugin api is out of date. Please run hack/update-generated-device-plugin.sh"
DEVICE_PLUGIN_ROOT="${KUBE_ROOT}/pkg/kubelet/apis/deviceplugin/v1alpha/"

source "${KUBE_ROOT}/hack/lib/protoc.sh"
kube::golang::setup_env

function cleanup {
	rm -rf ${DEVICE_PLUGIN_ROOT}/_tmp/
}

trap cleanup EXIT

mkdir -p ${DEVICE_PLUGIN_ROOT}/_tmp
cp ${DEVICE_PLUGIN_ROOT}/api.pb.go ${DEVICE_PLUGIN_ROOT}/_tmp/

KUBE_VERBOSE=3 "${KUBE_ROOT}/hack/update-generated-device-plugin.sh"
kube::protoc::diff "${DEVICE_PLUGIN_ROOT}/api.pb.go" "${DEVICE_PLUGIN_ROOT}/_tmp/api.pb.go" ${ERROR}
echo "Generated device plugin api is up to date."
