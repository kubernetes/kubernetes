#!/usr/bin/env bash

# Copyright 2024 The Kubernetes Authors.
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

# This script lints declarative validation comment tags on API resource files
# that have opted-in. It uses a specific golangci-lint configuration to invoke
# the kube-api-linter plugin for targeted checks.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

cd "${KUBE_ROOT}"

# The logic below is a replica of how hack/update-codegen.sh discovers files
# with '+k8s:validation-gen' tags, but simplified for this script.
# It finds all go files with the validation-gen tag, excluding vendor and testdata.
mapfile -t dirs < <(git grep --untracked -l '+k8s:validation-gen=' -- '**/*.go' ':!:*/testdata/*' ':!:vendor/*' | xargs -n1 dirname | sort -u)
if [[ ${#dirs[@]} -eq 0 ]]; then
  kube::log::status "No files with '+k8s:validation-gen' found to lint with kube-api-linter."
  exit 0
fi

packages=()
for dir in "${dirs[@]}"; do
  packages+=("./${dir}")
done

kube::log::status "Verifying API linting rules for packages with declarative validation using kube-api-linter..."
"${KUBE_ROOT}/hack/verify-golangci-lint.sh" -c "${KUBE_ROOT}/hack/kube-api-linter/validation-gen.golangci.yaml" "${packages[@]}"