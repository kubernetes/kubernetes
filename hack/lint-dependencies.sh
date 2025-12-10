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

# This script checks version dependencies of modules. It checks whether all
# pinned versions of checked dependencies match their preferred version or not.
# Usage: `hack/lint-dependencies.sh`.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

# Detect problematic GOPROXY settings that prevent lookup of dependencies
if [[ "${GOPROXY:-}" == "off" ]]; then
  kube::log::error "Cannot run with \$GOPROXY=off"
  exit 1
fi

kube::golang::setup_env
kube::util::require-jq

# Set the Go environment, otherwise we get "can't compute 'all' using the
# vendor directory".
export GOWORK=off
export GOFLAGS=-mod=mod

# let us log all errors before we exit
rc=0

# List of dependencies we need to avoid dragging back into kubernetes/kubernetes
# Check if unwanted dependencies are removed
# The array and map in `unwanted-dependencies.json` are in alphabetical order.
go run k8s.io/kubernetes/cmd/dependencyverifier "${KUBE_ROOT}/hack/unwanted-dependencies.json"

k8s_module_regex="k8s[.]io/(kubernetes"
for repo in $(kube::util::list_staging_repos); do
  k8s_module_regex="${k8s_module_regex}|${repo}"
done
k8s_module_regex="${k8s_module_regex})"

recursive_dependencies=$(go mod graph | grep -E " ${k8s_module_regex}" | grep -E -v "^${k8s_module_regex}" || true)
if [[ -n "${recursive_dependencies}" ]]; then
  echo "These external modules depend on k8s.io/kubernetes or staging modules, which is not allowed:"
  echo ""
  echo "${recursive_dependencies}"
fi

outdated=$(go list -m -json all | jq -r "
  select(.Replace.Version != null) |
  select(.Version != .Replace.Version) |
  select(.Path) |
  \"\(.Path)
    pinned:    \(.Replace.Version)
    preferred: \(.Version)
    hack/pin-dependency.sh \(.Path) \(.Version)\"
")
if [[ -n "${outdated}" ]]; then
  echo "These modules are pinned to versions different than the minimal preferred version."
  echo "That means that without replace directives, a different version would be selected,"
  echo "which breaks consumers of our published modules."
  echo "1. Use hack/pin-dependency.sh to switch to the preferred version for each module"
  echo "2. Run hack/update-vendor.sh to rebuild the vendor directory"
  echo "3. Run hack/lint-dependencies.sh to verify no additional changes are required"
  echo ""
  echo "${outdated}"
fi

noncanonical=$(go list -m -json all | jq -r "
  select(.Replace.Version != null) |
  select(.Path != .Replace.Path) |
  select(.Path) |
  \"  \(.Path) is replaced with \(.Replace.Path)\"
")
if [[ -n "${noncanonical}" ]]; then
  echo ""
  echo "These modules are pinned to non-canonical repos."
  echo "Revert to using the canonical repo for these modules before merge"
  echo ""
  echo "${noncanonical}"
fi

unused=$(comm -23 \
  <(go mod edit -json | jq -r '.Replace[] | select(.New.Version != null) | .Old.Path' | sort) \
  <(go list -m -json all | jq -r .Path | sort))
if [[ -n "${unused}" ]]; then
  echo ""
  echo "Use the given commands to remove pinned module versions that aren't actually used:"
  echo "${unused}" | xargs -L 1 echo 'go mod edit -dropreplace'
fi

if [[ -n "${unused}${outdated}${noncanonical}${recursive_dependencies}" ]]; then
  rc=1
else
  echo "All pinned versions of checked dependencies match their preferred version."
fi

exit $rc
