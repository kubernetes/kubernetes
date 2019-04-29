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
source "${KUBE_ROOT}/hack/lib/init.sh"

# Explicitly opt into go modules, even though we're inside a GOPATH directory
export GO111MODULE=on
# Explicitly clear GOFLAGS, since GOFLAGS=-mod=vendor breaks dependency resolution while rebuilding vendor
export GOFLAGS=
# Detect problematic GOPROXY settings that prevent lookup of dependencies
if [[ "${GOPROXY:-}" == "off" ]]; then
  kube::log::error "Cannot run with \$GOPROXY=off"
  exit 1
fi

kube::golang::verify_go_version
kube::util::require-jq

outdated=$(go list -m -json all | jq -r '
  select(.Replace.Version != null) | 
  select(.Version != .Replace.Version) | 
  "\(.Path)
    pinned:    \(.Replace.Version)
    preferred: \(.Version)
    hack/pin-dependency.sh \(.Path) \(.Version)"
')
if [[ -n "${outdated}" ]]; then
  echo "These modules are pinned to versions different than the minimal preferred version."
  echo "That means that without require directives, a different version would be selected."
  echo "The command to switch to the minimal preferred version is listed for each module."
  echo ""
  echo "${outdated}"
fi

unused=$(comm -23 \
  <(go mod edit -json | jq -r '.Replace[] | select(.New.Version != null) | .Old.Path' | sort) \
  <(go list -m -json all | jq -r .Path | sort))
if [[ -n "${unused}" ]]; then
  echo ""
  echo "Pinned module versions that aren't actually used:"
  echo "${unused}" | xargs -L 1 echo 'GO111MODULE=on go mod edit -dropreplace'
fi

if [[ -n "${unused}${outdated}" ]]; then
  exit 1
fi

echo "All pinned dependencies match their preferred version."
exit 0
