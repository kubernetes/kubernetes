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

KUBE_ROOT=${GOPATH}/src/k8s.io/kubernetes
BASE_PATH=k8s.io/kubernetes/staging/src/
BASE_PKG=k8s.io/sample-apiserver

source "${KUBE_ROOT}/hack/lib/init.sh"

clientgen="${PWD}/client-gen-binary"
listergen="${PWD}/lister-gen"
informergen="${PWD}/informer-gen"
# Register function to be called on EXIT to remove generated binary.
function cleanup {
  rm -f "${clientgen:-}"
  rm -f "${listergen:-}"
  rm -f "${informergen:-}"
}
trap cleanup EXIT

function generate_group() {
  local GROUP_NAME=$1
  local VERSION=$2
  local SERVER_BASE=${GOPATH}/src/${BASE_PATH}
  local CLIENT_PKG=${BASE_PKG}/pkg/client
  local LISTERS_PKG=${CLIENT_PKG}/listers_generated
  local INFORMERS_PKG=${CLIENT_PKG}/informers_generated
  local PREFIX=${BASE_PKG}/pkg/apis
  local INPUT_APIS=(
    ${GROUP_NAME}/
    ${GROUP_NAME}/${VERSION}
  )

  echo "Building client-gen"
  go build -o "${clientgen}" k8s.io/kubernetes/cmd/libs/go2idl/client-gen

  echo "generating clientset for group ${GROUP_NAME} and version ${VERSION} at ${GOPATH}/${BASE_PATH}${CLIENT_PKG}"
  ${clientgen} --input-base ${PREFIX} --input ${INPUT_APIS[@]} --clientset-path ${CLIENT_PKG}/clientset_generated --output-base=${GOPATH}/src/${BASE_PATH}
  ${clientgen} --clientset-name="clientset" --input-base ${PREFIX} --input ${GROUP_NAME}/${VERSION} --clientset-path ${CLIENT_PKG}/clientset_generated --output-base=${GOPATH}/src/${BASE_PATH}
  
  echo "Building lister-gen"
  go build -o "${listergen}" k8s.io/kubernetes/cmd/libs/go2idl/lister-gen

  echo "generating listers for group ${GROUP_NAME} and version ${VERSION} at ${GOPATH}/${BASE_PATH}${LISTERS_PKG}"
  ${listergen} --input-dirs ${BASE_PKG}/pkg/apis/wardle --input-dirs ${BASE_PKG}/pkg/apis/${GROUP_NAME}/${VERSION} --output-package ${LISTERS_PKG} --output-base ${SERVER_BASE}

  echo "Building informer-gen"
  go build -o "${informergen}" k8s.io/kubernetes/cmd/libs/go2idl/informer-gen

  echo "generating informers for group ${GROUP_NAME} and version ${VERSION} at ${GOPATH}/${BASE_PATH}${INFORMERS_PKG}"
  ${informergen} \
    --input-dirs ${BASE_PKG}/pkg/apis/${GROUP_NAME} --input-dirs ${BASE_PKG}/pkg/apis/${GROUP_NAME}/${VERSION} \
    --versioned-clientset-package ${CLIENT_PKG}/clientset_generated/clientset \
    --internal-clientset-package ${CLIENT_PKG}/clientset_generated/internalclientset \
    --listers-package ${LISTERS_PKG} \
    --output-package ${INFORMERS_PKG} \
    --output-base ${SERVER_BASE}
}

generate_group wardle v1alpha1
