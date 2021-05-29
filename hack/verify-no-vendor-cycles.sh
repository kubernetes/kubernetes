#!/usr/bin/env bash

# Copyright 2017 The Kubernetes Authors.
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

# This script checks whether packages under `vendor` directory have cyclic
# dependencies on `main` or `staging` repositories.
# Usage: `hack/verify-no-vendor-cycles.sh`.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

export GO111MODULE=auto

staging_repos=()
kube::util::read-array staging_repos < <(kube::util::list_staging_repos)
staging_repos_pattern=$(IFS="|"; echo "${staging_repos[*]}")

cd "${KUBE_ROOT}"

# Check for any module that is not main or staging and depends on main or staging
bad_deps=$(go mod graph | grep -vE "^k8s.io\/(kubernetes|${staging_repos_pattern})" | grep -E "\sk8s.io\/(kubernetes|${staging_repos_pattern})" || true)
if [[ -n "${bad_deps}" ]]; then
  echo "Found disallowed dependencies that transitively depend on k8s.io/kubernetes or staging modules:"
  echo "${bad_deps}"
  exit 1
fi

kube::util::ensure-temp-dir

# Get vendored packages dependencies
# Use -deps flag to include transitive dependencies
go list -mod=vendor -test -deps -json ./vendor/... > "${KUBE_TEMP}/deps.json"

# Check for any vendored package that imports main repo
# Staging repos are explicitly excluded even though go list does not currently consider symlinks
go run cmd/dependencycheck/dependencycheck.go -restrict "^k8s\.io/kubernetes/" -exclude "^k8s\.io/(${staging_repos_pattern})(/|$)" "${KUBE_TEMP}/deps.json"

# Check for any vendored package that imports a staging repo
# Staging repos are explicitly excluded even though go list does not currently consider symlinks
go run cmd/dependencycheck/dependencycheck.go -restrict "^k8s\.io/(${staging_repos_pattern})(/|$)" -exclude "^k8s\.io/(${staging_repos_pattern})(/|$)" "${KUBE_TEMP}/deps.json"
