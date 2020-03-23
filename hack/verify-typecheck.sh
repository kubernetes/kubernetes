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

# This script does a fast type check of kubernetes code for all platforms.
# Usage: `hack/verify-typecheck.sh`.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::golang::verify_go_version

cd "${KUBE_ROOT}"

make --no-print-directory -C "${KUBE_ROOT}" generated_files

ret=0
go run test/typecheck/main.go "$@" || ret=$?
if [[ $ret -ne 0 ]]; then
  echo "!!! Type Check has failed. This may cause cross platform build failures." >&2
  echo "!!! Please see https://git.k8s.io/kubernetes/test/typecheck for more information." >&2
  exit 1
fi
