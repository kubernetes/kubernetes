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

kube::golang::setup_env
kube::util::require-jq

BIN_PKGS=(
  # release binaries
  ./cmd/cloud-controller-manager
  ./cmd/kube-apiserver
  ./cmd/kube-controller-manager
  ./cmd/kube-proxy
  ./cmd/kube-scheduler
  ./cmd/kubectl
  ./cmd/kubectl-convert
  ./cmd/kubelet
  ./cmd/kubeadm
  # code generators
  ./staging/src/k8s.io/code-generator/cmd/*/
)

pkgs_with_testing_import=()
for pkg in "${BIN_PKGS[@]}"
do
  testing_deps="$(go list -json "${pkg}" | jq -r '[.Deps[] | select(endswith("testing")) ]| join(", ")')"
  if [ -n "${testing_deps}" ]; then
    pkgs_with_testing_import+=( "${pkg}: ${testing_deps}" )
  fi
done

if [ ${#pkgs_with_testing_import[@]} -ne 0 ]; then
  printf "%s\n" "Testing packages are imported in:"
  for pkg in "${pkgs_with_testing_import[@]}"; do
    printf "  %s\n" "${pkg}"
  done
  exit 1
fi

exit 0

