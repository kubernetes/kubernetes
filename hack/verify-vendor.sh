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

kube::golang::setup_env

# create a nice clean place to put our new vendor tree
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

  if ! _out="$(diff -Naupr -x "AUTHORS*" -x "CONTRIBUTORS*" vendor "${_kubetmp}/vendor")"; then
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

# Ensure we can tidy every repo using only its recorded versions
for repo in $(kube::util::list_staging_repos); do
  pushd "${_kubetmp}/staging/src/k8s.io/${repo}" >/dev/null 2>&1
    echo "Tidying k8s.io/${repo}..."
    GODEBUG=gocacheverify=1 go mod tidy
  popd >/dev/null 2>&1
done
pushd "${_kubetmp}" >/dev/null 2>&1
  echo "Tidying k8s.io/kubernetes..."
  GODEBUG=gocacheverify=1 go mod tidy
popd >/dev/null 2>&1

echo "Vendor Verified."
# ex: ts=2 sw=2 et filetype=sh
