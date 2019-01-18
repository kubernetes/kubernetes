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

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

readonly branch=${1:-${KUBE_VERIFY_GIT_BRANCH:-master}}
if ! [[ ${KUBE_FORCE_VERIFY_CHECKS:-} =~ ^[yY]$ ]] && \
  ! kube::util::has_changes_against_upstream_branch "${branch}" 'Godeps/' && \
  ! kube::util::has_changes_against_upstream_branch "${branch}" 'vendor/' && \
  ! kube::util::has_changes_against_upstream_branch "${branch}" 'hack/lib/' && \
  ! kube::util::has_changes_against_upstream_branch "${branch}" 'hack/.*godep'; then
  exit 0
fi

# Ensure we have the right godep version available
kube::util::ensure_godep_version

if [[ -z ${TMP_GOPATH:-} ]]; then
  # Create a nice clean place to put our new godeps
  _tmpdir="$(kube::realpath $(mktemp -d -t gopath.XXXXXX))"
else
  # reuse what we might have saved previously
  _tmpdir="${TMP_GOPATH}"
fi

if [[ -z ${KEEP_TMP:-} ]]; then
    KEEP_TMP=false
fi
function cleanup {
  if [ "${KEEP_TMP}" == "true" ]; then
    echo "Leaving ${_tmpdir} for you to examine or copy. Please delete it manually when finished. (rm -rf ${_tmpdir})"
  else
    echo "Removing ${_tmpdir}"
    rm -rf "${_tmpdir}"
  fi
}
trap cleanup EXIT

# Copy the contents of the kube directory into the nice clean place
_kubetmp="${_tmpdir}/src/k8s.io"
mkdir -p "${_kubetmp}"
# should create ${_kubectmp}/kubernetes
git archive --format=tar --prefix=kubernetes/ $(git write-tree) | (cd "${_kubetmp}" && tar xf -)
_kubetmp="${_kubetmp}/kubernetes"

# Do all our work in the new GOPATH
export GOPATH="${_tmpdir}"
export PATH="${GOPATH}/bin:${PATH}"

pushd "${_kubetmp}" > /dev/null 2>&1
  # Restore the Godeps into our temp directory
  hack/godep-restore.sh

  # Destroy deps in the copy of the kube tree
  rm -rf ./Godeps ./vendor

  # For some reason the kube tree needs to be a git repo for the godep tool to
  # run. Doesn't make sense.
  git init > /dev/null 2>&1

  # Recreate the Godeps using the nice clean set we just downloaded
  hack/godep-save.sh
popd > /dev/null 2>&1

ret=0

pushd "${KUBE_ROOT}" > /dev/null 2>&1
  # Test for diffs
  if ! _out="$(diff -Naupr --ignore-matching-lines='^\s*\"GoVersion\":' Godeps/Godeps.json ${_kubetmp}/Godeps/Godeps.json)"; then
    echo "Your Godeps.json is different:" >&2
    echo "${_out}" >&2
    echo "Godeps Verify failed." >&2
    echo "${_out}" > godepdiff.patch
    echo "If you're seeing this locally, run the below command to fix your Godeps.json:" >&2
    echo "patch -p0 < godepdiff.patch" >&2
    echo "(The above output can be saved as godepdiff.patch if you're not running this locally)" >&2
    echo "(The patch file should also be exported as a build artifact if run through CI)" >&2
    KEEP_TMP=true
    if [[ -f godepdiff.patch && -d "${ARTIFACTS:-}" ]]; then
      echo "Copying patch to artifacts.."
      cp godepdiff.patch "${ARTIFACTS:-}/"
    fi
    ret=1
  fi

  if ! _out="$(diff -Naupr -x "BUILD" -x "AUTHORS*" -x "CONTRIBUTORS*" vendor ${_kubetmp}/vendor)"; then
    echo "Your vendored results are different:" >&2
    echo "${_out}" >&2
    echo "Godeps Verify failed." >&2
    echo "${_out}" > vendordiff.patch
    echo "If you're seeing this locally, run the below command to fix your directories:" >&2
    echo "patch -p0 < vendordiff.patch" >&2
    echo "(The above output can be saved as godepdiff.patch if you're not running this locally)" >&2
    echo "(The patch file should also be exported as a build artifact if run through CI)" >&2
    KEEP_TMP=true
    if [[ -f vendordiff.patch && -d "${ARTIFACTS:-}" ]]; then
      echo "Copying patch to artifacts.."
      cp vendordiff.patch "${ARTIFACTS:-}/"
    fi
    ret=1
  fi
popd > /dev/null 2>&1

if [[ ${ret} -gt 0 ]]; then
  exit ${ret}
fi

echo "Godeps Verified."
# ex: ts=2 sw=2 et filetype=sh
