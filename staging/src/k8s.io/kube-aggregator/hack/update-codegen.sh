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

SCRIPT_ROOT=$(dirname "${BASH_SOURCE}")/..
SCRIPT_BASE=${SCRIPT_ROOT}/../..
CODEGEN_PKG=${CODEGEN_PKG:-$(cd ${SCRIPT_ROOT}; ls -d -1 ./vendor/k8s.io/code-generator 2>/dev/null || echo k8s.io/code-generator)}

if LANG=C sed --help 2>&1 | grep -q GNU; then
  SED="sed"
elif which gsed &>/dev/null; then
  SED="gsed"
else
  echo "Failed to find GNU sed as sed or gsed. If you are on Mac: brew install gnu-sed." >&2
  exit 1
fi

# Register function to be called on EXIT to remove generated binary.
function cleanup {
  rm -f "${CLIENTGEN:-}"
  rm -f "${listergen:-}"
  rm -f "${informergen:-}"
}
trap cleanup EXIT

echo "Building client-gen"
CLIENTGEN="${PWD}/client-gen"

go build -o "${CLIENTGEN}" ${CODEGEN_PKG}/cmd/client-gen

PREFIX=k8s.io/kube-aggregator/pkg/apis
INPUT_BASE="--input-base ${PREFIX}"
INPUT_APIS=(
apiregistration/
apiregistration/v1beta1
apiregistration/v1
)
INPUT="--input ${INPUT_APIS[@]}"
CLIENTSET_PATH="--output-package k8s.io/kube-aggregator/pkg/client/clientset_generated"

${CLIENTGEN} ${INPUT_BASE} ${INPUT} ${CLIENTSET_PATH} --output-base ${SCRIPT_BASE}
${CLIENTGEN} --clientset-name="clientset" ${INPUT_BASE} --input apiregistration/v1beta1,apiregistration/v1 ${CLIENTSET_PATH}  --output-base ${SCRIPT_BASE} --go-header-file ${SCRIPT_ROOT}/hack/boilerplate.go.txt


echo "Building lister-gen"
listergen="${PWD}/lister-gen"
go build -o "${listergen}" ${CODEGEN_PKG}/cmd/lister-gen

LISTER_INPUT="--input-dirs k8s.io/kube-aggregator/pkg/apis/apiregistration --input-dirs k8s.io/kube-aggregator/pkg/apis/apiregistration/v1beta1,k8s.io/kube-aggregator/pkg/apis/apiregistration/v1"
LISTER_PATH="--output-package k8s.io/kube-aggregator/pkg/client/listers"
${listergen} ${LISTER_INPUT} ${LISTER_PATH} --output-base ${SCRIPT_BASE} --go-header-file ${SCRIPT_ROOT}/hack/boilerplate.go.txt


echo "Building informer-gen"
informergen="${PWD}/informer-gen"
go build -o "${informergen}" ${CODEGEN_PKG}/cmd/informer-gen

${informergen} \
  --output-base ${SCRIPT_BASE} \
  --input-dirs k8s.io/kube-aggregator/pkg/apis/apiregistration --input-dirs k8s.io/kube-aggregator/pkg/apis/apiregistration/v1beta1,k8s.io/kube-aggregator/pkg/apis/apiregistration/v1 \
  --versioned-clientset-package k8s.io/kube-aggregator/pkg/client/clientset_generated/clientset \
  --internal-clientset-package k8s.io/kube-aggregator/pkg/client/clientset_generated/internalclientset \
  --listers-package k8s.io/kube-aggregator/pkg/client/listers \
  --output-package k8s.io/kube-aggregator/pkg/client/informers \
  --go-header-file ${SCRIPT_ROOT}/hack/boilerplate.go.txt
  "$@"
