#!/usr/bin/env bash

# Copyright 2018 The Kubernetes Authors.
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

# This script verifies the build without in-tree cloud providers.
# Usage: `hack/verify-typecheck-providerless.sh`.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..

cd "${KUBE_ROOT}"
# verify the providerless build
# https://github.com/kubernetes/enhancements/blob/master/keps/sig-cloud-provider/20190729-building-without-in-tree-providers.md
hack/verify-typecheck.sh --skip-test --tags=providerless --ignore-dirs=test
