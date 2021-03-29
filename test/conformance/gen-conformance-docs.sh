#!/usr/bin/env bash

# Copyright 2021 The Kubernetes Authors.
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

# This will canonicalize the path
KUBE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd -P)
cd "${KUBE_ROOT}"

# test sources -> json spec _output/specsummaries.json
test/conformance/gen-specsummaries.sh
# specsummaries.json -> conformance.md
test/conformance/spec-to-docs.sh
