#!/usr/bin/env bash

# Copyright 2022 The Kubernetes Authors.
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

# This script checks a PR for the coding style for the Go language files using
# golangci-lint. It does nothing when invoked as part of a normal "make
# verify".

set -o nounset
set -o pipefail

if [ ! "${PULL_NUMBER:-}" ]; then
  echo 'Not testing anything because this is not a pull request.'
  exit 0
fi

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..

"${KUBE_ROOT}/hack/verify-golangci-lint.sh" -r "${PULL_BASE_SHA}" -n
