#!/bin/bash

# Copyright 2015 The Kubernetes Authors All rights reserved.
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

# Conformance Tests a running Kubernetes cluster.
# Validates that the cluster was deployed, is accessible, and at least
# satisfies end-to-end tests marked as being a part of the conformance suite.
# Emphasis on broad coverage and being non-destructive over thoroughness.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..

TEST_ARGS="$@"

exec "${KUBE_ROOT}/hack/conformance-test.sh" ${TEST_ARGS}
