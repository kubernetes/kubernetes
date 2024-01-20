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

# This script verifies whether codes follow golang convention.
# Usage: `hack/verify-prerelease-lifecycle-tags.sh`.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

# Tags must be put on all non-alpha API types
# prerelease-lifecycle-gen itself makes sure every type with TypeMeta in the 
# package contains atleast an introduced tag

DIRGLOBS=(
  "staging/src/k8s.io/api/**/*/doc.go"
  "staging/src/k8s.io/kube-aggregator/pkg/apis/**/*/doc.go"
  "staging/src/k8s.io/apiextensions-apiserver/pkg/apis/**/*/doc.go"
)

cd "${KUBE_ROOT}"
for DOCGLOB in "${DIRGLOBS[@]}"; do
  if git --no-pager grep -L '// +k8s:prerelease-lifecycle-gen=true' -- "${DOCGLOB}" ":!*api*/*alpha*/doc.go"; then
    echo "!!! Some non-alpha packages doc.go do not include prerelease-lifecycle tags."
    echo "To fix these errors, add '// +k8s:prerelease-lifecycle-gen=true' to doc.go and"
    echo "add '// +k8s:prerelease-lifecycle-gen:introduced=1.<release>' to every type that embeds metav1.TypeMeta"
    exit 1
  fi
done
