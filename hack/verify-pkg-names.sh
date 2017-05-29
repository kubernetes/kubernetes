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

# Verify whether codes follow golang convention.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::golang::verify_go_version

cd "${KUBE_ROOT}"
if git --no-pager grep -E $'^(import |\t)[a-z]+[A-Z_][a-zA-Z]* "[^"]+"$' -- '**/*.go' ':(exclude)vendor/*' ':(exclude)staging/src/k8s.io/client-go/*vendor/*' ':(exclude)staging/src/k8s.io/metrics/*' ':(exclude)pkg/apis/admission/v1alpha1/zz_generated.conversion.go'; then
  echo "!!! Some package aliases break go conventions."
  echo "To fix these errors, do not use capitalized or underlined characters"
  echo "in pkg aliases. Refer to https://blog.golang.org/package-names for more info."
  exit 1
fi
