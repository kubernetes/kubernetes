#!/bin/bash

# Copyright 2015 The Kubernetes Authors All rights reserved.
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

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::golang::setup_env

# This list is kept in the order in which files ought to be processed. You must
# add new entries *after* any of their dependencies!
ordered_list=(\
  "./pkg/api/types.generated.go" \
  "./pkg/storage/testing/types.generated.go" \
  "./pkg/kubectl/testing/types.generated.go" \
  "./pkg/api/v1/types.generated.go" \
  "./pkg/apis/metrics/v1alpha1/types.generated.go" \
  "./pkg/apis/metrics/types.generated.go" \
  "./pkg/apis/extensions/v1beta1/types.generated.go" \
  "./pkg/apis/extensions/types.generated.go" \
  "./pkg/apiserver/testing/types.generated.go" \
  "./pkg/apis/componentconfig/types.generated.go" \
  "./pkg/apis/batch/v1/types.generated.go" \
  "./pkg/apis/batch/types.generated.go" \
  "./pkg/apis/autoscaling/v1/types.generated.go" \
  "./pkg/apis/autoscaling/types.generated.go" \
  "./pkg/apis/authorization/v1beta1/types.generated.go" \
  "./pkg/apis/authorization/types.generated.go" \
  "./pkg/apis/apps/v1alpha1/types.generated.go" \
  "./pkg/apis/apps/types.generated.go" \
  "./federation/apis/federation/v1alpha1/types.generated.go" \
  "./federation/apis/federation/types.generated.go" \
  "./cmd/libs/go2idl/client-gen/testdata/apis/testgroup.k8s.io/v1/types.generated.go" \
  "./cmd/libs/go2idl/client-gen/testdata/apis/testgroup.k8s.io/types.generated.go")

# To be helpful, find any new .generated.go files and tell users they need to
# add them to the above list.
found_files=($(
  find . -not \( \
      \( \
        -wholename './output' \
        -o -wholename './_output' \
        -o -wholename './release' \
        -o -wholename './target' \
        -o -wholename '*/third_party/*' \
        -o -wholename '*/Godeps/*' \
        -o -wholename '*/codecgen-*-1234.generated.go' \
      \) -prune \
    \) -name '*.generated.go' | sort -r))

# Sort, then check against the list of files we found to make sure that it
# hasn't changed.
readarray -t sorted_list < <(for a in "${ordered_list[@]}"; do echo "$a"; done | sort -r)
gCount=${#found_files[@]}
sCount=${#sorted_list[@]}
entriesToCheck=$(($gCount>$sCount?$gCount:$sCount))
for (( i=0; i<${entriesToCheck}; i++ )); do
  if [[ "${found_files[${i}]}" != "${sorted_list[${i}]}" ]]; then
    echo "found '${found_files[${i}]}', but expected '${sorted_list[${i}]}'."
    echo "You probably added or removed a types.go file. Please update hack/update-codecgen.sh"
    echo "and add or remove the file in the ordered_list variable."
    exit 1
  fi
done

# Register function to be called on EXIT to remove codecgen
# binary and also to touch the files that should be regenerated
# since they are first removed.
# This is necessary to make the script work after previous failure.
function cleanup {
  rm -f "${CODECGEN:-}"
  pushd "${KUBE_ROOT}" > /dev/null
  for generated_file in "${ordered_list[@]}"; do
    touch "${generated_file}" || true
  done
  popd > /dev/null
}
trap cleanup EXIT


CODECGEN="${PWD}/codecgen_binary"
godep go build -o "${CODECGEN}" github.com/ugorji/go/codec/codecgen

# Running codecgen fails if some of the files doesn't compile.
# Thus (since all the files are completely auto-generated and
# not required for the code to be compilable, we first remove
# them and the regenerate them.
for generated_file in "${ordered_list[@]}"; do
  rm -f "${generated_file}"
done

# Generate files in the dependency order.
for generated_file in "${ordered_list[@]}"; do
  initial_dir=${PWD}
  file=${generated_file//\.generated\.go/.go}
  echo "codecgen processing ${file}"
  # codecgen work only if invoked from directory where the file
  # is located.
  pushd "$(dirname ${file})" > /dev/null
  base_file=$(basename "${file}")
  base_generated_file=$(basename "${generated_file}")
  # We use '-d 1234' flag to have a deterministic output every time.
  # The constant was just randomly chosen.
  echo Running ${CODECGEN} -d 1234 -o  "${base_generated_file}" "${base_file}"
  ${CODECGEN} -d 1234 -o "${base_generated_file}" "${base_file}"
  # Add boilerplate at the beginning of the generated file.
  sed 's/YEAR/2015/' "${initial_dir}/hack/boilerplate/boilerplate.go.txt" > "${base_generated_file}.tmp"
  cat "${base_generated_file}" >> "${base_generated_file}.tmp"
  mv "${base_generated_file}.tmp" "${base_generated_file}"
  echo "${generated_file} is regenerated."
  popd > /dev/null
done
