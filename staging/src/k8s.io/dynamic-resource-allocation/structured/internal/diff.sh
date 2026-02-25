#!/usr/bin/env bash

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

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/../../../../../..

cd "${KUBE_ROOT}" || exit 1

function diff-allocators () {
    a="$1"
    b="$2"

    for i in allocator pools; do
        diff --ignore-matching-lines="^package \\($a\\|$b\\)\$" -c "staging/src/k8s.io/dynamic-resource-allocation/structured/internal/${a}/${i}_${a}.go" "staging/src/k8s.io/dynamic-resource-allocation/structured/internal/${b}/${i}_${b}.go"
    done
}

diff-allocators stable incubating
diff-allocators incubating experimental