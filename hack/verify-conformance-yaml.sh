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

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
cd "${KUBE_ROOT}"

# generate json spec -> yaml
test/conformance/gen-conformance-yaml.sh

# diff generated and checked-in
if diff -u test/conformance/testdata/conformance.yaml _output/conformance.yaml; then
  echo PASS
  exit 0
fi
echo 'See instructions in test/conformance/README.md'
exit 1
