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

SCRIPT_ROOT=$(dirname ${BASH_SOURCE})/..
CODEGEN_PKG=${CODEGEN_PKG:-$(cd ${SCRIPT_ROOT}; ls -d -1 ./vendor/k8s.io/code-generator 2>/dev/null || echo ../code-generator)}

CLIENTSET_NAME_VERSIONED=clientset \
CLIENTSET_PKG_NAME=clientset_generated \
${CODEGEN_PKG}/generate-groups.sh deepcopy,client,lister,informer \
  k8s.io/kube-aggregator/pkg/client k8s.io/kube-aggregator/pkg/apis \
  "apiregistration:v1beta1,v1" \
  --output-base "$(dirname ${BASH_SOURCE})/../../.." \
  --go-header-file ${SCRIPT_ROOT}/hack/boilerplate.go.txt

CLIENTSET_NAME_VERSIONED=clientset \
CLIENTSET_PKG_NAME=clientset_generated \
CLIENTSET_NAME_INTERNAL=internalclientset \
${CODEGEN_PKG}/generate-internal-groups.sh deepcopy,client,lister,informer,conversion \
  k8s.io/kube-aggregator/pkg/client k8s.io/kube-aggregator/pkg/apis k8s.io/kube-aggregator/pkg/apis \
  "apiregistration:v1beta1,v1" \
  --output-base "$(dirname ${BASH_SOURCE})/../../.." \
  --go-header-file ${SCRIPT_ROOT}/hack/boilerplate.go.txt
