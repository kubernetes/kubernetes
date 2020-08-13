#!/usr/bin/env bash

# Copyright 2014 The Kubernetes Authors.
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

# This script checks coding style for go language files in each
# Kubernetes package by golint.
# Usage: `hack/verify-golint.sh`.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"
source "${KUBE_ROOT}/hack/lib/util.sh"

kube::golang::verify_go_version


# Ensure that we find the binaries we build before anything else.
export GOBIN="${KUBE_OUTPUT_BINPATH}"
PATH="${GOBIN}:${PATH}"

# Install golint
echo 'installing golint'
pushd "${KUBE_ROOT}/hack/tools" >/dev/null
  GO111MODULE=on go install golang.org/x/lint/golint
popd >/dev/null


cd "${KUBE_ROOT}"

# Check that the file is in alphabetical order
failure_file="${KUBE_ROOT}/hack/.golint_failures"
kube::util::check-file-in-alphabetical-order "${failure_file}"

export IFS=$'\n'
# NOTE: when "go list -e ./..." is run within GOPATH, it turns the k8s.io/kubernetes
# as the prefix, however if we run it outside it returns the full path of the file
# with a leading underscore. We'll need to support both scenarios for all_packages.
all_packages=()
while IFS='' read -r line; do all_packages+=("$line"); done < <(go list -e ./... | grep -vE "/(third_party|vendor|staging/src/k8s.io/client-go/pkg|generated|clientset_generated)" | sed -e 's|^k8s.io/kubernetes/||' -e "s|^_\(${KUBE_ROOT}/\)\{0,1\}||")
# The regex below removes any "#" character and anything behind it and including any
# whitespace before it. Then it removes empty lines.
failing_packages=()
while IFS='' read -r line; do failing_packages+=("$line"); done < <(sed -e 's/[[:blank:]]*#.*//' -e '/^$/d' "$failure_file")
unset IFS
errors=()
not_failing=()
for p in "${all_packages[@]}"; do
  # Run golint on package/*.go file explicitly to validate all go files
  # and not just the ones for the current platform. This also will ensure that
  # _test.go files are linted.
  # Generated files are ignored, and each file is passed through golint
  # individually, as if one file in the package contains a fatal error (such as
  # a foo package with a corresponding foo_test package), golint seems to choke
  # completely.
  # Ref: https://github.com/kubernetes/kubernetes/pull/67675
  # Ref: https://github.com/golang/lint/issues/68
  failedLint=$(find "$p"/*.go | grep -vE "(zz_generated.*.go|generated.pb.go|generated.proto|types_swagger_doc_generated.go)" | xargs -L1 golint 2>/dev/null)
  kube::util::array_contains "$p" "${failing_packages[@]}" && in_failing=$? || in_failing=$?
  if [[ -n "${failedLint}" ]] && [[ "${in_failing}" -ne "0" ]]; then
    errors+=( "${failedLint}" )
  fi
  if [[ -z "${failedLint}" ]] && [[ "${in_failing}" -eq "0" ]]; then
    not_failing+=( "$p" )
  fi
done

# Check that all failing_packages actually still exist
gone=()
for p in "${failing_packages[@]}"; do
  kube::util::array_contains "$p" "${all_packages[@]}" || gone+=( "$p" )
done

# Check to be sure all the packages that should pass lint are.
if [ ${#errors[@]} -eq 0 ]; then
  echo 'Congratulations!  All Go source files have been linted.'
else
  {
    echo "Errors from golint:"
    for err in "${errors[@]}"; do
      echo "$err"
    done
    echo
    echo 'Please review the above warnings. You can test via "golint" and commit the result.'
    echo 'If the above warnings do not make sense, you can exempt this package from golint'
    echo 'checking by adding it to hack/.golint_failures (if your reviewer is okay with it).'
    echo
  } >&2
  exit 1
fi

if [[ ${#not_failing[@]} -gt 0 ]]; then
  {
    echo "Some packages in hack/.golint_failures are passing golint. Please remove them."
    echo
    for p in "${not_failing[@]}"; do
      echo "  $p"
    done
    echo
  } >&2
  exit 1
fi

if [[ ${#gone[@]} -gt 0 ]]; then
  {
    echo "Some packages in hack/.golint_failures do not exist anymore. Please remove them."
    echo
    for p in "${gone[@]}"; do
      echo "  $p"
    done
    echo
  } >&2
  exit 1
fi
