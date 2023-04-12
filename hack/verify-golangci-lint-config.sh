#!/usr/bin/env bash

# Copyright 2023 The Kubernetes Authors.
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

# This script checks that golangci-strict.yaml and golangci.yaml remain in
# sync. Lines that are intentionally different must have a comment which
# mentions golangci.yaml or golangci-lint.yaml.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..

if differences=$(diff --context --ignore-blank-lines \
                      --ignore-matching-lines='^ *#' \
                      --ignore-matching-lines='#.*golangci\(-strict\)*.yaml' \
                      "${KUBE_ROOT}/hack/golangci.yaml" "${KUBE_ROOT}/hack/golangci-strict.yaml" ); then
    echo "hack/golangci.yaml and hack/golangci-strict.yaml are synchronized."
else
    cat >&2 <<EOF
Unexpected differences between hack/golangci.yaml and hack/golangci-strict.yaml:

${differences}

If these differences are intentional, then add comments at the end of each
different line in both files that mention golangci-strict.yaml or
golangci.yaml.
EOF
    exit 1
fi
