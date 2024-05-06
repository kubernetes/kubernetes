#!/usr/bin/env bash

# Copyright 2023 The Kubernetes Authors.
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

# This script checks whether the testing.init symbol is present in any
# of the release binaries and fails if it finds one. This check is needed
# to avoid including test libraries in production binaries as they often lack
# rigorous review and sufficient testing.
# Usage: `hack/verify-test-code.sh`.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"
cd "${KUBE_ROOT}"

RELEASE_BIN_PKGS=(
  "${KUBE_ROOT}/cmd/cloud-controller-manager"
  "${KUBE_ROOT}/cmd/kube-apiserver"
  "${KUBE_ROOT}/cmd/kube-controller-manager"
  "${KUBE_ROOT}/cmd/kube-proxy"
  "${KUBE_ROOT}/cmd/kube-scheduler"
  "${KUBE_ROOT}/cmd/kubectl"
  "${KUBE_ROOT}/cmd/kubectl-convert"
  "${KUBE_ROOT}/cmd/kubelet"
  "${KUBE_ROOT}/cmd/kubeadm"
)

pkgs_with_testing_import=()
for file in "${RELEASE_BIN_PKGS[@]}"
do
  if [ "$(go list -json "${file}" | jq 'any(.Deps[]; . == "testing")')" == "true" ]
  then
    pkgs_with_testing_import+=( "${file}" )
  fi
done

if [ ${#pkgs_with_testing_import[@]} -ne 0 ]; then
  printf "%s\n" "Testing package imported in:"
  for file in "${pkgs_with_testing_import[@]}"; do
    printf "\t%s\n" "${file}"
  done
  exit 1
fi

exit 0

