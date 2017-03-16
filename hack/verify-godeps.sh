#!/bin/bash

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

# As of go 1.6, the vendor experiment is enabled by default.
export GO15VENDOREXPERIMENT=1

#### HACK ####
# Sometimes godep just can't handle things. This lets use manually put
# some deps in place first, so godep won't fall over.
preload-dep() {
  org="$1"
  project="$2"
  sha="$3"
  # project_dir ($4) is optional, if unset we will generate it
  if [[ -z ${4:-} ]]; then
    project_dir="${GOPATH}/src/${org}/${project}.git"
  else
    project_dir="${4}"
  fi

  echo "**HACK** preloading dep for ${org} ${project} at ${sha} into ${project_dir}"
  git clone "https://${org}/${project}" "${project_dir}" > /dev/null 2>&1
  pushd "${project_dir}" > /dev/null
    git checkout "${sha}"
  popd > /dev/null
}

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

readonly branch=${1:-${KUBE_VERIFY_GIT_BRANCH:-master}}
if ! [[ ${KUBE_FORCE_VERIFY_CHECKS:-} =~ ^[yY]$ ]] && \
  ! kube::util::has_changes_against_upstream_branch "${branch}" 'Godeps/' && \
  ! kube::util::has_changes_against_upstream_branch "${branch}" 'vendor/'; then
  exit 0
fi

if [[ -z ${TMP_GOPATH:-} ]]; then
  # Create a nice clean place to put our new godeps
  _tmpdir="$(mktemp -d -t gopath.XXXXXX)"
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
  export GODEP=""
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

pushd "${_kubetmp}" 2>&1 > /dev/null
  kube::util::ensure_godep_version v79

  export GOPATH="${GOPATH}:${_kubetmp}/staging"
  # Fill out that nice clean place with the kube godeps
  echo "Starting to download all kubernetes godeps. This takes a while"
  godep restore
  echo "Download finished"

  # Destroy deps in the copy of the kube tree
  rm -rf ./Godeps ./vendor

  # For some reason the kube tree needs to be a git repo for the godep tool to
  # run. Doesn't make sense.
  git init > /dev/null 2>&1

  # Recreate the Godeps using the nice clean set we just downloaded
  hack/godep-save.sh
popd 2>&1 > /dev/null

ret=0

pushd "${KUBE_ROOT}" 2>&1 > /dev/null
  # Test for diffs
  if ! _out="$(diff -Naupr --ignore-matching-lines='^\s*\"GoVersion\":' --ignore-matching-line='^\s*\"GodepVersion\":' --ignore-matching-lines='^\s*\"Comment\":' Godeps/Godeps.json ${_kubetmp}/Godeps/Godeps.json)"; then
    echo "Your Godeps.json is different:"
    echo "${_out}"
    echo "Godeps Verify failed."
    echo "${_out}" > godepdiff.patch
    echo "If you're seeing this locally, run the below command to fix your Godeps.json:"
    echo "patch -p0 < godepdiff.patch"
    echo "(The above output can be saved as godepdiff.patch if you're not running this locally)"
    KEEP_TMP=true
    ret=1
  fi

  if ! _out="$(diff -Naupr -x "BUILD" -x "AUTHORS*" -x "CONTRIBUTORS*" vendor ${_kubetmp}/vendor)"; then
    echo "Your vendored results are different:"
    echo "${_out}"
    echo "Godeps Verify failed."
    echo "${_out}" > vendordiff.patch
    echo "If you're seeing this locally, run the below command to fix your directories:"
    echo "patch -p0 < vendordiff.patch"
    echo "(The above output can be saved as godepdiff.patch if you're not running this locally)"
    KEEP_TMP=true
    ret=1
  fi
popd 2>&1 > /dev/null

if [[ ${ret} > 0 ]]; then
  exit ${ret}
fi

echo "Godeps Verified."
# ex: ts=2 sw=2 et filetype=sh
