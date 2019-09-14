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
KUBELET_PLUGIN_REGISTRATION_ROOT="${KUBE_ROOT}/pkg/kubelet/apis/pluginregistration"
KUBELET_PLUGIN_REGISTRATION_VERSION="$(find "${KUBELET_PLUGIN_REGISTRATION_ROOT}" -maxdepth 1 -type d -path "${KUBELET_PLUGIN_REGISTRATION_ROOT}/*" -printf "%f\n")"

source "${KUBE_ROOT}/hack/lib/protoc.sh"
kube::golang::setup_env

function cleanup {
    for VERSION in ${KUBELET_PLUGIN_REGISTRATION_VERSION}
    do
        rm -rf "${KUBELET_PLUGIN_REGISTRATION_ROOT}/${VERSION}/_tmp"
    done
}

trap cleanup EXIT

for VERSION in ${KUBELET_PLUGIN_REGISTRATION_VERSION}
do
    KUBELET_PLUGIN_REGISTRATION_DIR="${KUBELET_PLUGIN_REGISTRATION_ROOT}/${VERSION}"

    mkdir -p "${KUBELET_PLUGIN_REGISTRATION_DIR}/_tmp"
    cp "${KUBELET_PLUGIN_REGISTRATION_DIR}/api.pb.go" "${KUBELET_PLUGIN_REGISTRATION_DIR}/_tmp/"

    KUBE_VERBOSE=3 "${KUBE_ROOT}/hack/update-generated-kubelet-plugin-registration.sh"
    kube::protoc::diff "${KUBELET_PLUGIN_REGISTRATION_DIR}/api.pb.go" "${KUBELET_PLUGIN_REGISTRATION_DIR}/_tmp/api.pb.go" "${ERROR}"
    rm -rf "${KUBELET_PLUGIN_REGISTRATION_DIR}/_tmp"
    echo "Generated Kubelet Plugin Registration ${VERSION} api is up to date."
done
