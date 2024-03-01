#!/usr/bin/env bash

# Copyright 2020 The Kubernetes Authors.
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

# These are "internal" modules.  For various reasons, we want them to be
# decoupled from their parent modules.
MODULES=(
    hack/tools
    staging/src/k8s.io/code-generator/examples
    staging/src/k8s.io/kms/internal/plugins/_mock
)

# Detect problematic GOPROXY settings that prevent lookup of dependencies
if [[ "${GOPROXY:-}" == "off" ]]; then
  kube::log::error "Cannot run hack/update-internal-modules.sh with \$GOPROXY=off"
  exit 1
fi

kube::golang::setup_env

for mod in "${MODULES[@]}"; do
  pushd "${KUBE_ROOT}/${mod}" >/dev/null
    echo "=== tidying go.mod/go.sum in ${mod}"
    go mod edit -fmt
    go mod tidy
  popd >/dev/null
done
