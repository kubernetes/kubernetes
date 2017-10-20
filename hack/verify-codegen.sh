#!/bin/bash

# Copyright 2014 The Kubernetes Authors.
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
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::golang::setup_env

# call verify on sub-project for now
#
# Note: these must be before the main script call because the later calls the sub-project's
#       update-codegen.sh scripts. We wouldn't see any error on changes then.
CODEGEN_PKG=./vendor/k8s.io/code-generator vendor/k8s.io/code-generator/hack/verify-codegen.sh
CODEGEN_PKG=./vendor/k8s.io/code-generator vendor/k8s.io/kube-aggregator/hack/verify-codegen.sh
CODEGEN_PKG=./vendor/k8s.io/code-generator vendor/k8s.io/sample-apiserver/hack/verify-codegen.sh
CODEGEN_PKG=./vendor/k8s.io/code-generator vendor/k8s.io/sample-controller/hack/verify-codegen.sh
CODEGEN_PKG=./vendor/k8s.io/code-generator vendor/k8s.io/apiextensions-apiserver/hack/verify-codegen.sh
CODEGEN_PKG=./vendor/k8s.io/code-generator vendor/k8s.io/metrics/hack/verify-codegen.sh

"${KUBE_ROOT}/hack/update-codegen.sh" --verify-only
