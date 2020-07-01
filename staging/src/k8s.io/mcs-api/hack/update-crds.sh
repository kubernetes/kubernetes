#!/usr/bin/env bash

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

SCRIPT_ROOT=$(dirname "${BASH_SOURCE[0]}")/..

CRD_ROOT="${SCRIPT_ROOT}/config/crds"
CONTROLLER_GEN="go run sigs.k8s.io/controller-tools/cmd/controller-gen"
CRD_OUTPUT=${CRD_OUTPUT:-"${CRD_ROOT}"}

${CONTROLLER_GEN} paths="./pkg/apis/..." schemapatch:manifests="${CRD_ROOT}" output:dir="${CRD_OUTPUT}"

