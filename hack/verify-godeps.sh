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
  project="$1"
  sha="$2"

  echo "**HACK** preloading dep ${project} at ${sha}"
  ./build/run.sh hack/run-in-gopath.sh go get "${project}"
  ./build/run.sh hack/run-in-gopath.sh bash -c 'git checkout -C "${GOPATH}/src/${project}" "${sha}"'
}

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

readonly branch=${1:-${KUBE_VERIFY_GIT_BRANCH:-master}}
if ! [[ ${KUBE_FORCE_VERIFY_CHECKS:-} =~ ^[yY]$ ]] && \
  ! kube::util::has_changes_against_upstream_branch "${branch}" 'Godeps/' && \
  ! kube::util::has_changes_against_upstream_branch "${branch}" 'vendor/' && \
  ! kube::util::has_changes_against_upstream_branch "${branch}" 'hack/.*godep'; then
  exit 0
fi

export KUBE_RUN_COPY_OUTPUT=N

# Restore the Godeps
hack/godep-restore.sh

# Recreate the Godeps using the nice clean set we just downloaded
hack/godep-save.sh

ret=0

pushd "${KUBE_ROOT}" 2>&1 > /dev/null
  # Test for diffs
  if ! _out="$(git diff Godeps/Godeps.json)"; then
    echo "Your Godeps.json is different:"
    echo "${_out}" > godepdiff.patch
    cat godepdiff.patch
    echo "Godeps Verify failed."
    echo "If you're seeing this locally, run the below command to fix your Godeps.json:"
    echo "patch -p1 < godepdiff.patch"
    echo "(The above output can be saved as godepdiff.patch if you're not running this locally)"
    echo "(The patch file should also be exported as a build artifact if run through CI)"
    KEEP_TMP=true
    if [[ -f godepdiff.patch && -d "${ARTIFACTS_DIR:-}" ]]; then
      echo "Copying patch to artifacts.."
      cp godepdiff.patch "${ARTIFACTS_DIR:-}/"
    fi
    ret=1
  fi

  if ! _out="$(git diff vendor)"; then
    echo "Your vendored results are different:"
    echo "${_out}" > vendordiff.patch
    cat vendordiff.patch
    echo "Godeps Verify failed."
    echo "If you're seeing this locally, run the below command to fix your directories:"
    echo "patch -p1 < vendordiff.patch"
    echo "(The above output can be saved as godepdiff.patch if you're not running this locally)"
    echo "(The patch file should also be exported as a build artifact if run through CI)"
    KEEP_TMP=true
    if [[ -f vendordiff.patch && -d "${ARTIFACTS_DIR:-}" ]]; then
      echo "Copying patch to artifacts.."
      cp vendordiff.patch "${ARTIFACTS_DIR:-}/"
    fi
    ret=1
  fi
popd 2>&1 > /dev/null

if [[ ${ret} > 0 ]]; then
  exit ${ret}
fi

echo "Godeps Verified."
# ex: ts=2 sw=2 et filetype=sh
