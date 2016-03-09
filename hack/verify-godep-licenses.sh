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

KUBE_ROOT="$(cd "$(dirname "${BASH_SOURCE}")/.." && pwd -P)"

branch="${1:-master}"
# notice this uses ... to find the first shared ancestor
if ! git diff origin/"${branch}"...HEAD | grep 'Godeps/' > /dev/null; then
  exit 0
fi

# create a nice clean place to put our new godeps
# must be in the user dir (e.g. KUBE_ROOT) in order for the docker volume mount to work with docker-machine on macs
_tmpdir="$(mktemp -d "${KUBE_ROOT}/kube-godep-licenses.XXXXXX")"
echo "Created workspace: ${_tmpdir}"
function cleanup {
  echo "Removing workspace: ${_tmpdir}"
  rm -rf "${_tmpdir}"
}
trap cleanup EXIT

cp -r "${KUBE_ROOT}/LICENSE" "${_tmpdir}/"
cp -r "${KUBE_ROOT}/Godeps" "${_tmpdir}/Godeps"

# Update Godep Licenses
KUBE_ROOT="${_tmpdir}" "${KUBE_ROOT}/hack/update-godep-licenses.sh"

# Compare Godep Licenses
if ! _out="$(diff -Naupr ${KUBE_ROOT}/Godeps/LICENSES ${_tmpdir}/Godeps/LICENSES)"; then
  echo "Your godep licenses file is out of date. Run hack/update-godep-licenses.sh and commit the results."
  echo "${_out}"
  exit 1
fi
