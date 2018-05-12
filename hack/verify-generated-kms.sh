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

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
KUBE_KMS_GRPC_ROOT="${KUBE_ROOT}/staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1/"
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::golang::setup_env

function cleanup {
	rm -rf ${KUBE_KMS_GRPC_ROOT}/_tmp/
}

trap cleanup EXIT

mkdir -p ${KUBE_KMS_GRPC_ROOT}/_tmp
cp ${KUBE_KMS_GRPC_ROOT}/service.pb.go ${KUBE_KMS_GRPC_ROOT}/_tmp/

ret=0
KUBE_VERBOSE=3 "${KUBE_ROOT}/hack/update-generated-kms.sh"
diff -I "gzipped FileDescriptorProto" -I "0x" -Naupr ${KUBE_KMS_GRPC_ROOT}/_tmp/service.pb.go ${KUBE_KMS_GRPC_ROOT}/service.pb.go || ret=$?
if [[ $ret -eq 0 ]]; then
    echo "Generated KMS gRPC is up to date."
    cp ${KUBE_KMS_GRPC_ROOT}/_tmp/service.pb.go ${KUBE_KMS_GRPC_ROOT}/
else
    echo "Generated KMS gRPC is out of date. Please run hack/update-generated-kms.sh"
    exit 1
fi
