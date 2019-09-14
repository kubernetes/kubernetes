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

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
ERROR="Device plugin api is out of date. Please run hack/update-generated-device-plugin.sh"

DEVICE_PLUGIN_ROOT="${KUBE_ROOT}/pkg/kubelet/apis/deviceplugin"
DEVICE_PLUGIN_VERSION="$(find "${DEVICE_PLUGIN_ROOT}" -maxdepth 1 -type d -path "${DEVICE_PLUGIN_ROOT}/*" -printf "%f\n")"

source "${KUBE_ROOT}/hack/lib/protoc.sh"
kube::golang::setup_env

function cleanup {
    for VERSION in ${DEVICE_PLUGIN_VERSION}
    do
        rm -rf "${DEVICE_PLUGIN_ROOT}/${VERSION}/_tmp"
    done
}

trap cleanup EXIT

for VERSION in ${DEVICE_PLUGIN_VERSION}
do
    DEVICE_PLUGIN_DIR="${DEVICE_PLUGIN_ROOT}/${VERSION}"

    mkdir -p "${DEVICE_PLUGIN_DIR}/_tmp"
    cp "${DEVICE_PLUGIN_DIR}/api.pb.go" "${DEVICE_PLUGIN_DIR}/_tmp/"

    KUBE_VERBOSE=3 "${KUBE_ROOT}/hack/update-generated-device-plugin.sh"
    kube::protoc::diff "${DEVICE_PLUGIN_DIR}/api.pb.go" "${DEVICE_PLUGIN_DIR}/_tmp/api.pb.go" "${ERROR}"
    rm -rf "${DEVICE_PLUGIN_DIR}/_tmp"

    echo "Generated device plugin ${VERSION} api is up to date."
done
