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
source "${KUBE_ROOT}/hack/lib/init.sh"

# Explicitly opt into go modules, even though we're inside a GOPATH directory
export GO111MODULE=on

kube::util::ensure_clean_working_dir
# This sets up the environment, like GOCACHE, which keeps the worktree cleaner.
kube::golang::setup_env

_tmpdir="$(kube::realpath "$(mktemp -d -t "$(basename "$0").XXXXXX")")"
git worktree add -f -q "${_tmpdir}" HEAD
kube::util::trap_add "git worktree remove -f ${_tmpdir}" EXIT
cd "${_tmpdir}"

# Update the mocks in ${_tmpdir}
hack/update-mocks.sh

# Test for diffs
diffs=$(git status --porcelain | wc -l)
if [[ ${diffs} -gt 0 ]]; then
  echo "Mock files are out of date" >&2
  git diff
  echo "Please run 'hack/update-mocks.sh'" >&2
  exit 1
fi
