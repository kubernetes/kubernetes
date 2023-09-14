#!/usr/bin/env bash

# Copyright 2020 The Kubernetes Authors.
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

kube::util::ensure_clean_working_dir
# This sets up the environment, like GOCACHE, which keeps the worktree cleaner.
kube::golang::setup_env

_tmpdir="$(kube::realpath "$(mktemp -d -t verify-internal-modules.XXXXXX)")"
kube::util::trap_add "rm -rf ${_tmpdir:?}" EXIT

_tmp_gopath="${_tmpdir}/go"
_tmp_kuberoot="${_tmp_gopath}/src/k8s.io/kubernetes"
git worktree add -f "${_tmp_kuberoot}" HEAD
kube::util::trap_add "git worktree remove -f ${_tmp_kuberoot}" EXIT

pushd "${_tmp_kuberoot}" >/dev/null
./hack/update-internal-modules.sh
popd

git -C "${_tmp_kuberoot}" add -N .
diff=$(git -C "${_tmp_kuberoot}" diff HEAD || true)

if [[ -n "${diff}" ]]; then
  echo "${diff}" >&2
  echo >&2
  echo "Run ./hack/update-internal-modules.sh" >&2
  exit 1
fi
