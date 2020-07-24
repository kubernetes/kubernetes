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

# This script checks whether updating of Pod resources API is needed or not. We
# should run `hack/update-generated-pod-resources.sh` if Pod resources API is
# out of date.
# Usage: `hack/verify-generated-pod-resources.sh`.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
POD_RESOURCES_ALPHA="${KUBE_ROOT}/pkg/kubelet/apis/podresources/v1alpha1/"
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::golang::setup_env

function cleanup {
	rm -rf "${POD_RESOURCES_ALPHA}/_tmp/"
}

trap cleanup EXIT

mkdir -p "${POD_RESOURCES_ALPHA}/_tmp"
cp "${POD_RESOURCES_ALPHA}/api.pb.go" "${POD_RESOURCES_ALPHA}/_tmp/"

ret=0
KUBE_VERBOSE=3 "${KUBE_ROOT}/hack/update-generated-pod-resources.sh"
diff -I "gzipped FileDescriptorProto" -I "0x" -Naupr "${POD_RESOURCES_ALPHA}/_tmp/api.pb.go" "${POD_RESOURCES_ALPHA}/api.pb.go" || ret=$?
if [[ $ret -eq 0 ]]; then
    echo "Generated pod resources api is up to date."
    cp "${POD_RESOURCES_ALPHA}/_tmp/api.pb.go" "${POD_RESOURCES_ALPHA}/"
else
    echo "Generated pod resources api is out of date. Please run hack/update-generated-pod-resources.sh"
    exit 1
fi
