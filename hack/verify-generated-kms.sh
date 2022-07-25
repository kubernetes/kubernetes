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

# This script checks whether updating of KMS gRPC is needed or not. We should
# run `hack/update-generated-kms.sh` if KMS gRPC is out of date.
# Usage: `hack/verify-generated-kms.sh`.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
ERROR="KMS gRPC is out of date. Please run hack/update-generated-kms.sh"
KUBE_KMS_V1BETA1="${KUBE_ROOT}/staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1/"
KUBE_KMS_V2ALPHA1="${KUBE_ROOT}/staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v2alpha1/"

source "${KUBE_ROOT}/hack/lib/protoc.sh"
kube::golang::setup_env

function cleanup {
	rm -rf "${KUBE_KMS_V1BETA1}/_tmp/"
	rm -rf "${KUBE_KMS_V2ALPHA1}/_tmp/"
}

trap cleanup EXIT

mkdir -p "${KUBE_KMS_V1BETA1}/_tmp"
cp "${KUBE_KMS_V1BETA1}/api.pb.go" "${KUBE_KMS_V1BETA1}/_tmp/"
mkdir -p "${KUBE_KMS_V2ALPHA1}/_tmp"
cp "${KUBE_KMS_V2ALPHA1}/api.pb.go" "${KUBE_KMS_V2ALPHA1}/_tmp/"

KUBE_VERBOSE=3 "${KUBE_ROOT}/hack/update-generated-kms.sh"
kube::protoc::diff "${KUBE_KMS_V1BETA1}/api.pb.go" "${KUBE_KMS_V1BETA1}/_tmp/api.pb.go" "${ERROR}"
echo "Generated kms v1beta1 api is up to date."
kube::protoc::diff "${KUBE_KMS_V2ALPHA1}/api.pb.go" "${KUBE_KMS_V2ALPHA1}/_tmp/api.pb.go" "${ERROR}"
echo "Generated kms v2alpha1 api is up to date."
