#!/usr/bin/env bash

# Copyright 2021 The Kubernetes Authors.
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

# This script generates mock files using mockgen.
# Usage: `hack/update-mocks.sh`.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"
# Explicitly opt into go modules, even though we're inside a GOPATH directory
export GO111MODULE=on

_tmp="${KUBE_ROOT}/_tmp_build_tag_files"
mkdir -p "${_tmp}"

function cleanup {
	rm -rf "$_tmp"
	rm -f "tempfile"
}

trap cleanup EXIT

kube::golang::verify_go_version

echo 'installing mockgen'
pushd "${KUBE_ROOT}/hack/tools" >/dev/null
  go install github.com/golang/mock/mockgen
popd >/dev/null

find_files() {
  find . -not \( \
      \( \
        -wholename './output' \
        -o -wholename './.git' \
        -o -wholename './_output' \
        -o -wholename './_gopath' \
        -o -wholename './release' \
        -o -wholename './target' \
        -o -wholename '*/third_party/*' \
        -o -wholename '*/vendor/*' \
        -o -wholename './staging/src/k8s.io/client-go/*vendor/*' \
        -o -wholename '*/bindata.go' \
      \) -prune \
    \) -name '*.go'
}

cd "${KUBE_ROOT}"

echo 'executing go generate command on below files'

for IFILE in $(find_files | xargs grep --files-with-matches -e '//go:generate mockgen'); do
  temp_file_name=$(mktemp --tmpdir="${_tmp}")
  # serach for build tag used in file
  build_tag_string=$(grep -o '+build.*$' "$IFILE") || true
  
  # if the file does not have build string
  if [ -n "$build_tag_string" ]
  then
    # write the build tag in the temp file
    echo -n "$build_tag_string" > "$temp_file_name"
    
    # if +build tag is defined in interface file
    BUILD_TAG_FILE=$temp_file_name go generate -v "$IFILE"
  else
    # if no +build tag is defined in interface file
    go generate -v "$IFILE"
  fi
done


# get the changed mock files
files=$(git diff --name-only)
for file in $files; do
  if [ "$file" == "hack/update-mocks.sh" ]; then
    continue
  fi

  # serach for build tags used in file
  # //go:build !providerless
  # // +build !providerless
  go_build_tag_string=$(grep -o 'go:build.*$' "$file") || true
  build_tag_string=$(grep -o '+build.*$' "$file") || true
  new_header=''

  # if the file has both headers
  if [ -n "$build_tag_string" ] && [ -n "$go_build_tag_string" ]
  then
    # create a new header with the build string and the copyright text
    new_header=$(echo -e "//""$go_build_tag_string""\n""//" "$build_tag_string""\n" | cat - hack/boilerplate/boilerplate.generatego.txt)

    # ignore the first line (build tag) from the file
    tail -n +3 "$file" > tempfile
  fi

  # if the file has only // +build !providerless header
  if [ -n "$build_tag_string" ] && [ -z "$go_build_tag_string" ]
  then
    # create a new header with the build string and the copyright text
    new_header=$(echo -e "//" "$build_tag_string""\n" | cat - hack/boilerplate/boilerplate.generatego.txt)

    # ignore the first line (build tag) from the file
    tail -n +2 "$file" > tempfile
  fi

  # if the file has only //go:build !providerless header
  if [ -z "$build_tag_string" ] && [ -n "$go_build_tag_string" ]
  then
    # create a new header with the build string and the copyright text
    new_header=$(echo -e "//""$go_build_tag_string""\n" | cat - hack/boilerplate/boilerplate.generatego.txt)

    # ignore the first line (build tag) from the file
    tail -n +2 "$file" > tempfile
  fi

  # if the header if generted
  if [ -n "$new_header" ]
  then
    # write the newly generated header file to the original file
    echo -e "$new_header" | cat - tempfile > "$file"
  else
    # if no build string insert at the top
    cat hack/boilerplate/boilerplate.generatego.txt "$file" > tempfile && \
    mv tempfile "$file"
  fi
done
