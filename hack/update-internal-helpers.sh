#!/usr/bin/env bash

# Copyright The Kubernetes Authors.
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
source "${KUBE_ROOT}/hack/lib/init.sh"

cd "${KUBE_ROOT}"

echo "Generating internal helpers..."

# Build the generator
cd "${KUBE_ROOT}/hack/tools"
go install ./internal-helper-gen

cd "${KUBE_ROOT}"

# Define directories to translate: [input_dir]:[output_dir]
HELPERS=(
  "staging/src/k8s.io/component-helpers/resource:pkg/apis/core/helper/resource"
  "pkg/apis/core/v1/helper/qos:pkg/apis/core/helper/qos"
  "pkg/api/v1/pod:pkg/api/pod"
  "pkg/api/v1/service:pkg/api/service"
)

for entry in "${HELPERS[@]}"; do
  IFS=":" read -r input output <<< "${entry}"
  internal-helper-gen -input="${KUBE_ROOT}/${input}" -output="${KUBE_ROOT}/${output}"
  
  echo "Running goimports on ${output}..."
  go -C "${KUBE_ROOT}/hack/tools" run golang.org/x/tools/cmd/goimports -w "${KUBE_ROOT}/${output}"

  echo "Verifying generated code builds and passes tests..."
  go test "${KUBE_ROOT}/${output}"
done
