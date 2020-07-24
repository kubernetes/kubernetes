#!/usr/bin/env bash

# Copyright 2016 The Kubernetes Authors.
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

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/../..
source "${KUBE_ROOT}/hack/lib/init.sh"

cd "${KUBE_ROOT}"

# If called directly, exit.
if [[ "${CALLED_FROM_MAIN_MAKEFILE:-""}" == "" ]]; then
    echo "ERROR: $0 should not be run directly." >&2
    echo >&2
    echo "Please run this command using \"make vet\""
    exit 1
fi

# This is required before we run govet for the results to be correct.
# See https://github.com/golang/go/issues/16086 for details.
go install ./cmd/...

# Filter out arguments that start with "-" and move them to goflags.
targets=()
for arg; do
  if [[ "${arg}" == -* ]]; then
    goflags+=("${arg}")
  else
    targets+=("${arg}")
  fi
done

if [[ ${#targets[@]} -eq 0 ]]; then
  # Do not run on third_party directories or generated client code or build tools.
  while IFS='' read -r line; do
    targets+=("${line}")
  done < <(go list -e ./... | grep -E -v "/(build|third_party|vendor|staging|clientset_generated|hack)/")
fi

go vet "${goflags[@]:+${goflags[@]}}" "${targets[@]}"
