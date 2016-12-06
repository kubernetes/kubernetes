#!/bin/bash

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

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../../..
APIFEDERATOR_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

# Register function to be called on EXIT to remove generated binary.
function cleanup {
  rm -f "${CLIENTGEN:-}"
  rm -f "${listergen:-}"
  rm -f "${informergen:-}"
}
trap cleanup EXIT

echo "Building client-gen"
CLIENTGEN="${PWD}/client-gen-binary"
go build -o "${CLIENTGEN}" ./cmd/libs/go2idl/client-gen

PREFIX=k8s.io/kubernetes/cmd/kubernetes-discovery/pkg/apis
INPUT_BASE="--input-base ${PREFIX}"
INPUT_APIS=(
apiregistration/
apiregistration/v1alpha1
)
INPUT="--input ${INPUT_APIS[@]}"
CLIENTSET_PATH="--clientset-path k8s.io/kubernetes/cmd/kubernetes-discovery/pkg/client/clientset_generated"

${CLIENTGEN} ${INPUT_BASE} ${INPUT} ${CLIENTSET_PATH} 
${CLIENTGEN} --clientset-name="release_1_5" ${INPUT_BASE} --input apiregistration/v1alpha1 ${CLIENTSET_PATH} 


echo "Building lister-gen"
listergen="${PWD}/lister-gen"
go build -o "${listergen}" ./cmd/libs/go2idl/lister-gen

LISTER_INPUT="--input-dirs k8s.io/kubernetes/cmd/kubernetes-discovery/pkg/apis/apiregistration --input-dirs k8s.io/kubernetes/cmd/kubernetes-discovery/pkg/apis/apiregistration/v1alpha1"
LISTER_PATH="--output-package k8s.io/kubernetes/cmd/kubernetes-discovery/pkg/client/listers"
${listergen} ${LISTER_INPUT} ${LISTER_PATH}


echo "Building informer-gen"
informergen="${PWD}/informer-gen"
go build -o "${informergen}" ./cmd/libs/go2idl/informer-gen

${informergen} \
  --input-dirs k8s.io/kubernetes/cmd/kubernetes-discovery/pkg/apis/apiregistration --input-dirs k8s.io/kubernetes/cmd/kubernetes-discovery/pkg/apis/apiregistration/v1alpha1 \
  --versioned-clientset-package k8s.io/kubernetes/cmd/kubernetes-discovery/pkg/client/clientset_generated/release_1_5 \
  --internal-clientset-package k8s.io/kubernetes/cmd/kubernetes-discovery/pkg/client/clientset_generated/internalclientset \
  --listers-package k8s.io/kubernetes/cmd/kubernetes-discovery/pkg/client/listers \
  --output-package k8s.io/kubernetes/cmd/kubernetes-discovery/pkg/client/informers
  "$@"
