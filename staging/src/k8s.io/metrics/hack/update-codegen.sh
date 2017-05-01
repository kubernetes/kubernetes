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

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../../../../..
METRICS_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

# Register function to be called on EXIT to remove generated binary.
function cleanup {
  rm -f "${CLIENTGEN:-}"
}
trap cleanup EXIT

echo "Building client-gen"
CLIENTGEN="${PWD}/client-gen-binary"
go build -o "${CLIENTGEN}" ./cmd/libs/go2idl/client-gen

PREFIX=k8s.io/metrics/pkg/apis
INPUT_BASE="--input-base ${PREFIX}"
CLIENTSET_PATH="--clientset-path k8s.io/metrics/pkg/client/clientset_generated"

${CLIENTGEN} --clientset-name="clientset" ${INPUT_BASE} --input metrics/v1alpha1 ${CLIENTSET_PATH} --output-base ${KUBE_ROOT}/vendor

# we skip informers and listers for metrics, because we don't quite support the requisite operations yet
# we skip generating the internal clientset as it's not really needed
