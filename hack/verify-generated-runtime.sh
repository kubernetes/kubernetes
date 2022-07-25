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

# This script checks whether updating of container runtime API is needed or not.
# We should run `hack/update-generated-runtime.sh` if container runtime API is
# out of date.
# Usage: `hack/verify-generated-runtime.sh`.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
KUBE_REMOTE_RUNTIME_ROOT="${KUBE_ROOT}/staging/src/k8s.io/cri-api/pkg/apis/runtime/"
source "${KUBE_ROOT}/hack/lib/init.sh"

runtime_versions=("v1alpha2" "v1")

kube::golang::setup_env

function cleanup {
	for v in "${runtime_versions[@]}"; do
		rm -rf "${KUBE_REMOTE_RUNTIME_ROOT}/${v}/_tmp/"
	done
}

trap cleanup EXIT

function verify_generated_code() {
	RUNTIME_API_VERSION="$1"
	KUBE_REMOTE_RUNTIME_PATH="${KUBE_REMOTE_RUNTIME_ROOT}/${RUNTIME_API_VERSION}"
	mkdir -p "${KUBE_REMOTE_RUNTIME_PATH}/_tmp"
	cp "${KUBE_REMOTE_RUNTIME_PATH}/api.pb.go" "${KUBE_REMOTE_RUNTIME_PATH}/_tmp/"

	ret=0
	KUBE_VERBOSE=3 "${KUBE_ROOT}/hack/update-generated-runtime.sh"
	diff -I "gzipped FileDescriptorProto" -I "0x" -Naupr "${KUBE_REMOTE_RUNTIME_PATH}/_tmp/api.pb.go" "${KUBE_REMOTE_RUNTIME_PATH}/api.pb.go" || ret=$?
	if [[ $ret -eq 0 ]]; then
	    echo "Generated container runtime api is up to date."
	    cp "${KUBE_REMOTE_RUNTIME_PATH}/_tmp/api.pb.go" "${KUBE_REMOTE_RUNTIME_PATH}/"
	else
	    echo "Generated container runtime api is out of date. Please run hack/update-generated-runtime.sh"
	    exit 1
	fi
}

for v in "${runtime_versions[@]}"; do
	verify_generated_code "${v}"
done
