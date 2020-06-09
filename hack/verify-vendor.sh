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

# This script checks whether fixing of vendor directory or go.mod is needed or
# not. We should run `hack/update-vendor.sh` if actually fixes them.
# Usage: `hack/verify-vendor.sh`.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

# create a nice clean place to put our new vendor tree
# must be in the user dir (e.g. KUBE_ROOT) in order for the docker volume mount
# to work with docker-machine on macs
mkdir -p "${KUBE_ROOT}/_tmp"
_tmpdir="$(mktemp -d "${KUBE_ROOT}/_tmp/kube-vendor.XXXXXX")"

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
kube::util::trap_add cleanup EXIT

# Copy the contents of the kube directory into the nice clean place (which is NOT shaped like a GOPATH)
_kubetmp="${_tmpdir}/kubernetes"
mkdir -p "${_kubetmp}"
tar --exclude=.git --exclude="./_*" -c . | (cd "${_kubetmp}" && tar xf -)

# Do all our work in module mode
export GO111MODULE=on

pushd "${_kubetmp}" > /dev/null 2>&1
  # Destroy deps in the copy of the kube tree
  rm -rf ./vendor ./LICENSES

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

  # Verify we are pinned to matching levels
  hack/lint-dependencies.sh >&2
popd > /dev/null 2>&1

if [[ ${ret} -gt 0 ]]; then
  exit ${ret}
fi

echo "Vendor Verified."
# ex: ts=2 sw=2 et filetype=sh
