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

export KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

if [[ ! -f "${KUBE_ROOT}/vendor/BUILD" ]]; then
  echo "${KUBE_ROOT}/vendor/BUILD does not exist." >&2
  echo >&2
  echo "Run ./hack/update-bazel.sh" >&2
  exit 1
fi

# Remove generated files prior to running kazel.
# TODO(spxtr): Remove this line once Bazel is the only way to build.
rm -f "${KUBE_ROOT}/pkg/generated/openapi/zz_generated.openapi.go"

_tmpdir="$(kube::realpath $(mktemp -d -t verify-bazel.XXXXXX))"
kube::util::trap_add "rm -rf ${_tmpdir}" EXIT

_tmp_gopath="${_tmpdir}/go"
_tmp_kuberoot="${_tmp_gopath}/src/k8s.io/kubernetes"
mkdir -p "${_tmp_kuberoot}/.."
cp -a "${KUBE_ROOT}" "${_tmp_kuberoot}/.."

cd "${_tmp_kuberoot}"
GOPATH="${_tmp_gopath}" PATH="${_tmp_gopath}/bin:${PATH}" ./hack/update-bazel.sh

diff=$(diff -Naupr -x '_output' "${KUBE_ROOT}" "${_tmp_kuberoot}" || true)

if [[ -n "${diff}" ]]; then
  echo "${diff}" >&2
  echo >&2
  echo "Run ./hack/update-bazel.sh" >&2
  exit 1
fi
