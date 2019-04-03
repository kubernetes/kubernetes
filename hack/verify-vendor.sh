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
  ! kube::util::has_changes "${branch}" 'go.mod' && \
  ! kube::util::has_changes "${branch}" 'go.sum' && \
  ! kube::util::has_changes "${branch}" 'vendor/' && \
  ! kube::util::has_changes "${branch}" 'staging/' && \
  ! kube::util::has_changes "${branch}" 'hack/lib/' && \
  ! kube::util::has_changes "${branch}" 'hack/.*vendor'; then
  exit 0
fi

if [[ -z ${TMP_GOPATH:-} ]]; then
  # Create a nice clean place to put our new vendor
  _tmpdir="$(kube::realpath "$(mktemp -d -t verifyvendor.XXXXXX)")"
else
  # reuse what we might have saved previously
  _tmpdir="${TMP_GOPATH}"
fi

if [[ -z ${KEEP_TMP:-} ]]; then
    KEEP_TMP=false
fi
function cleanup {
  # make go module dirs writeable
  chmod -R +w "${_tmpdir}"
  if [ "${KEEP_TMP}" == "true" ]; then
    echo "Leaving ${_tmpdir} for you to examine or copy. Please delete it manually when finished. (rm -rf ${_tmpdir})"
  else
    echo "Removing ${_tmpdir}"
    rm -rf "${_tmpdir}"
  fi
}
trap cleanup EXIT

# Copy the contents of the kube directory into the nice clean place (which is NOT shaped like a GOPATH)
_kubetmp="${_tmpdir}"
mkdir -p "${_kubetmp}"
# should create ${_kubectmp}/kubernetes
git archive --format=tar --prefix=kubernetes/ "$(git write-tree)" | (cd "${_kubetmp}" && tar xf -)
_kubetmp="${_kubetmp}/kubernetes"

# Do all our work with an unset GOPATH
export GOPATH=

pushd "${_kubetmp}" > /dev/null 2>&1
  # Destroy deps in the copy of the kube tree
  rm -rf ./Godeps/LICENSES ./vendor

  # Recreate the vendor tree using the nice clean set we just downloaded
  hack/update-vendor.sh
popd > /dev/null 2>&1

ret=0

pushd "${KUBE_ROOT}" > /dev/null 2>&1
  # Test for diffs
  if ! _out="$(diff -Naupr --ignore-matching-lines='^\s*\"GoVersion\":' go.mod "${_kubetmp}/go.mod")"; then
    echo "Your go.mod file is different:" >&2
    echo "${_out}" >&2
    echo "Vendor Verify failed." >&2
    echo "If you're seeing this locally, run the below command to fix your go.mod:" >&2
    echo "hack/update-vendor.sh" >&2
    ret=1
  fi

  if ! _out="$(diff -Naupr -x "BUILD" -x "AUTHORS*" -x "CONTRIBUTORS*" vendor "${_kubetmp}/vendor")"; then
    echo "Your vendored results are different:" >&2
    echo "${_out}" >&2
    echo "Vendor Verify failed." >&2
    echo "${_out}" > vendordiff.patch
    echo "If you're seeing this locally, run the below command to fix your directories:" >&2
    echo "hack/update-vendor.sh" >&2
    ret=1
  fi
popd > /dev/null 2>&1

if [[ ${ret} -gt 0 ]]; then
  exit ${ret}
fi

echo "Vendor Verified."
# ex: ts=2 sw=2 et filetype=sh
