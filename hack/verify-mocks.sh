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

# This script checks whether updating of Mock files generated from Interfaces
# is needed or not. We should run `hack/update-mocks.sh` 
# if Mock files are out of date.
# Usage: `hack/verify-mocks.sh`.


set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
export KUBE_ROOT
source "${KUBE_ROOT}/hack/lib/init.sh"

# Explicitly opt into go modules, even though we're inside a GOPATH directory
export GO111MODULE=on

_tmp="${KUBE_ROOT}/_tmp"
mkdir -p "${_tmp}"

"${KUBE_ROOT}/hack/update-mocks.sh"

# If there are untracked generated mock files which needed to be committed
if git_status=$(git status --porcelain --untracked=normal 2>/dev/null) && [[ -n "${git_status}" ]]; then
  echo "!!! You may have untracked mock files."
fi

# get the changed mock files
files=$(git diff --name-only)

# copy the changed files to the _tmp directory for later comparison
mock_files=()
for file in $files; do
  if [ "$file" == "hack/verify-mocks.sh" ]; then
    continue
  fi

  # create the directory in _tmp
  mkdir -p "$_tmp/""$(dirname "$file")"
  cp "$file" "$_tmp/""$file"
  mock_files+=("$file")

  # reset the current file
  git checkout "$file"
done

echo "diffing process started for ${#mock_files[@]} files"
ret=0
for file in "${mock_files[@]}"; do
  diff -Naupr -B \
    -I '^/\*' \
    -I 'Copyright The Kubernetes Authors.' \
    -I 'Licensed under the Apache License, Version 2.0 (the "License");' \
    -I 'you may not use this file except in compliance with the License.' \
    -I 'You may obtain a copy of the License at' \
    -I 'http://www.apache.org/licenses/LICENSE-2.0' \
    -I 'Unless required by applicable law or agreed to in writing, software' \
    -I 'distributed under the License is distributed on an "AS IS" BASIS,' \
    -I 'WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.' \
    -I 'See the License for the specific language governing permissions and' \
    -I 'limitations under the License.' \
    -I '^\*/' \
    "$file" "$_tmp/""$file" || ret=$?

  if [[ $ret -ne 0 ]]; then
    echo "Mock files are out of date. Please run hack/update-mocks.sh" >&2
    exit 1
  fi
done
echo "up to date"
