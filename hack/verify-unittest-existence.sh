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

# This checks existence of corresponding unit test for go modules.

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/util.sh"
cd "${KUBE_ROOT}"

# Check that the file is in alphabetical order
failure_file="${KUBE_ROOT}/hack/.unittest_failures"
kube::util::check-file-in-alphabetical-order "${failure_file}"

# NOTE: when "go list -e ./..." is run within GOPATH, it turns the k8s.io/kubernetes
# as the prefix, however if we run it outside it returns the full path of the file
# with a leading underscore. We'll need to support both scenarios for all_packages.
all_packages=(
  $(go list -e ./... | egrep -v "/(third_party|vendor|staging/src/k8s.io/client-go/pkg|generated|clientset_generated)" | sed -e 's|^k8s.io/kubernetes/||' -e "s|^_\(${KUBE_ROOT}/\)\{0,1\}||")
)
failing_packages=(
  $(cat $failure_file)
)

errors=()
not_failing=()

for path in "${all_packages[@]}"; do
  kube::util::array_contains "$path" "${failing_packages[@]}" && in_failing=$? || in_failing=$?
  non_existent=0

  files=$(ls "$path"/*.go | egrep -v "_test\.go$" | egrep -v "(zz_generated.*.go|generated.pb.go|generated.proto|types_swagger_doc_generated.go)")
  for file in $files;  do
    testfile="${file//.go/_test.go}"
    if [ ! -f "${testfile}" ]; then
      non_existent=1
    fi
  done
  if [[ "${non_existent}" -eq "1" ]] && [[ "${in_failing}" -ne "0" ]]; then
    errors+=( ${path} )
  fi
  if [[ "${non_existent}" -eq "0" ]] && [[ "${in_failing}" -eq "0" ]]; then
    not_failing+=( ${path} )
  fi
done

# Check to be sure all the packages that should have corresponding unit test modules.
if [ ${#errors[@]} -eq 0 ]; then
  echo 'Congratulations!  All Go source files have corresponding unit test modules.'
else
  {
    echo "The following go source files don't have corresponding unit test modules:"
    for err in "${errors[@]}"; do
      echo "$err"
    done
    echo
    echo 'Please review the above warnings.'
    echo 'If the above warnings do not make sense, you can exempt this package from unit test'
    echo 'checking by adding it to hack/.unittest_failures (if your reviewer is okay with it).'
    echo
  } >&2
  false
fi

if [[ ${#not_failing[@]} -gt 0 ]]; then
  {
    echo "Some packages in hack/.unittest_failures are passing golint. Please remove them."
    echo
    for p in "${not_failing[@]}"; do
      echo "  $p"
    done
    echo
  } >&2
  false
fi

