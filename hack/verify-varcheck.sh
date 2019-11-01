#!/usr/bin/env bash
# Copyright 2019 The Kubernetes Authors.
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

# Install golangci-lint
"${KUBE_ROOT}"/hack/install-golangci-lint.sh

# varcheck checking
while read -r dir; do
    if [ "$dir" = "${KUBE_ROOT}/cmd" ];then
        continue
    fi
    golangci-lint run -v "${dir}" --disable-all -E varcheck --timeout 5m
done < <(find "${KUBE_ROOT}/cmd" -maxdepth 1 -type d) 
