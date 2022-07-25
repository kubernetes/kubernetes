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
KUBE_KMS_V1BETA1="${KUBE_ROOT}/staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1/"
KUBE_KMS_V2ALPHA1="${KUBE_ROOT}/staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v2alpha1/"

source "${KUBE_ROOT}/hack/lib/protoc.sh"
kube::protoc::generate_proto "${KUBE_KMS_V1BETA1}"
kube::protoc::generate_proto "${KUBE_KMS_V2ALPHA1}"
