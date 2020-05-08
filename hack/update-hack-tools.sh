#!/usr/bin/env bash

# Copyright 2020 The Kubernetes Authors.
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
source "${KUBE_ROOT}/hack/lib/init.sh"

# Explicitly opt into go modules, even though we're inside a GOPATH directory
export GO111MODULE=on

# Detect problematic GOPROXY settings that prevent lookup of dependencies
if [[ "${GOPROXY:-}" == "off" ]]; then
  kube::log::error "Cannot run hack/update-hack-tools.sh with \$GOPROXY=off"
  exit 1
fi

kube::golang::verify_go_version

pushd "${KUBE_ROOT}/hack/tools" >/dev/null
  echo "=== tidying go.mod/go.sum in hack/tools"
  go mod edit -fmt
  go mod tidy
  go mod vendor

  LICENSE_ROOT="${PWD}" "${KUBE_ROOT}/hack/update-vendor-licenses.sh"
  rm -rf vendor
popd >/dev/null
