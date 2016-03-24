#!/bin/bash

# Copyright 2016 The Kubernetes Authors All rights reserved.
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

# Verify that E2E's use Describe wrappers, so that we can auto tag and provide
# other wrapper functionality for the entire suite.
set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
cd "${KUBE_ROOT}"

findDescViolations() {
    find ./test/e2e/ -name '*.go' | xargs cat | grep "\sDescribe("
}

# There should be only one call to describe.
if [ $(findDescViolations | wc -l) != 1 ]; then
    echo "The following lines use Describe instead of KubeDescribe."
    echo "Describe() is a reserved term which only should called via the KubeDescribe wrapper function."
    findDescViolations
fi
