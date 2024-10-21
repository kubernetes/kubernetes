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

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"
source "${KUBE_ROOT}/hack/lib/util.sh"

# make sure everything is committed
kube::util::ensure_clean_working_dir

# This sets up the environment, like GOCACHE, which keeps the worktree cleaner.
kube::golang::setup_env

go install golang.org/x/vuln/cmd/govulncheck@v1.1.2

# KUBE_VERIFY_GIT_BRANCH is populated in verify CI jobs
BRANCH="${KUBE_VERIFY_GIT_BRANCH:-master}"

kube::util::ensure-temp-dir
WORKTREE="${KUBE_TEMP}/worktree"

# Create a copy of the repo with $BRANCH checked out
git worktree add -f "${WORKTREE}" "${BRANCH}"
# Clean up the copy on exit
kube::util::trap_add "git worktree remove -f ${WORKTREE}" EXIT

govulncheck -scan package ./... > "${KUBE_TEMP}/head.txt" || true
pushd "${WORKTREE}" >/dev/null
  govulncheck -scan package ./... > "${KUBE_TEMP}/pr-base.txt" || true
popd >/dev/null

echo -e "\n HEAD: $(cat "${KUBE_TEMP}"/head.txt)" 
echo -e "\n PR_BASE: $(cat "${KUBE_TEMP}/pr-base.txt")" 

diff -s -u --ignore-all-space "${KUBE_TEMP}"/pr-base.txt "${KUBE_TEMP}"/head.txt || true
