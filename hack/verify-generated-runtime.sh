#!/usr/bin/env bash

# Copyright 2016 The Kubernetes Authors.
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
KUBE_REMOTE_RUNTIME_ROOT="${KUBE_ROOT}/pkg/kubelet/apis/cri/runtime/v1alpha2"
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::golang::setup_env

function cleanup {
	rm -rf ${KUBE_REMOTE_RUNTIME_ROOT}/_tmp/
}

trap cleanup EXIT

mkdir -p ${KUBE_REMOTE_RUNTIME_ROOT}/_tmp
cp ${KUBE_REMOTE_RUNTIME_ROOT}/api.pb.go ${KUBE_REMOTE_RUNTIME_ROOT}/_tmp/

ret=0
KUBE_VERBOSE=3 "${KUBE_ROOT}/hack/update-generated-runtime.sh"
diff -I "gzipped FileDescriptorProto" -I "0x" -Naupr ${KUBE_REMOTE_RUNTIME_ROOT}/_tmp/api.pb.go ${KUBE_REMOTE_RUNTIME_ROOT}/api.pb.go || ret=$?
if [[ $ret -eq 0 ]]; then
    echo "Generated container runtime api is up to date."
    cp ${KUBE_REMOTE_RUNTIME_ROOT}/_tmp/api.pb.go ${KUBE_REMOTE_RUNTIME_ROOT}/
else
    echo "Generated container runtime api is out of date. Please run hack/update-generated-runtime.sh"
    exit 1
fi
