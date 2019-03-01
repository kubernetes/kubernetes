#!/usr/bin/env bash

# Copyright 2015 The Kubernetes Authors.
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

readonly branch=${1:-${KUBE_VERIFY_GIT_BRANCH:-master}}
if ! [[ ${KUBE_FORCE_VERIFY_CHECKS:-} =~ ^[yY]$ ]] && \
  ! kube::util::has_changes "${branch}" 'Godeps/' && \
  ! kube::util::has_changes "${branch}" 'vendor/'; then
  exit 0
fi

# create a nice clean place to put our new godeps
# must be in the user dir (e.g. KUBE_ROOT) in order for the docker volume mount
# to work with docker-machine on macs
mkdir -p "${KUBE_ROOT}/_tmp"
_tmpdir="$(mktemp -d "${KUBE_ROOT}/_tmp/kube-godep-licenses.XXXXXX")"
#echo "Created workspace: ${_tmpdir}"
function cleanup {
  #echo "Removing workspace: ${_tmpdir}"
  rm -rf "${_tmpdir}"
}
trap cleanup EXIT

cp -r "${KUBE_ROOT}/Godeps" "${_tmpdir}/Godeps"
ln -s "${KUBE_ROOT}/LICENSE" "${_tmpdir}"
ln -s "${KUBE_ROOT}/vendor" "${_tmpdir}"

# Update Godep Licenses
LICENSE_ROOT="${_tmpdir}" "${KUBE_ROOT}/hack/update-godep-licenses.sh"

# Compare Godep Licenses
if ! _out="$(diff -Naupr "${KUBE_ROOT}/Godeps/LICENSES" "${_tmpdir}/Godeps/LICENSES")"; then
  echo "Your godep licenses file is out of date. Run hack/update-godep-licenses.sh and commit the results." >&2
  echo "${_out}" >&2
  exit 1
fi
