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
  ! kube::util::has_changes_against_upstream_branch "${branch}" 'staging/' && \
  ! kube::util::has_changes_against_upstream_branch "${branch}" 'vendor/' && \
  ! kube::util::has_changes_against_upstream_branch "${branch}" 'hack/lib/' && \
  ! kube::util::has_changes_against_upstream_branch "${branch}" 'hack/.*godep'; then
  exit 0
fi

# Ensure we have the right kdep version available
kube::util::ensure_kdep_version

if [[ -z ${TMP_GOPATH:-} ]]; then
  # Create a nice clean place to put our new deps
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
  # Destroy deps in the copy of the kube tree
  rm -rf ./vendor

  # Recreate the deps. kdep has been installed above, and anyway we don't have
  # the sources anymore.
  SKIP_KDEP_INSTALL=true ./hack/dep-ensure.sh
popd > /dev/null 2>&1

ret=0

pushd "${KUBE_ROOT}" > /dev/null 2>&1
  # Test for diffs in generated Gopkg.lock
  GOPKG_FILE="Gopkg.lock"
  if ! _out="$(diff -Naupr ${GOPKG_FILE} ${_kubetmp}/${GOPKG_FILE})"; then
    echo "Your ${GOPKG_FILE} is different:" >&2
    echo "${_out}" >&2
    echo "Deps Verify failed." >&2
    echo "${_out}" > gopkg.patch
    echo "If you're seeing this locally, run the below command to fix your Gopkg.lock:" >&2
    echo "patch -p0 < gopkg.patch" >&2
    echo "(The above output can be saved as gopkg.patch if you're not running this locally)" >&2
    echo "(The patch file should also be exported as a build artifact if run through CI)" >&2
    KEEP_TMP=true
    if [[ -f gopkg.patch && -d "${ARTIFACTS_DIR:-}" ]]; then
        echo "Copying patch to artifacts.."
        cp "gopkg.patch" "${ARTIFACTS_DIR:-}/"
    fi
    ret=1
  fi

  # Test for diffs in generated Godeps.json
  for GODEPSDIR in Godeps `find staging -type d -name "Godeps"`; do
    BASEDIR=`dirname $GODEPSDIR`
    GODEPS_FILE="$GODEPSDIR/Godeps.json"
    if ! _out="$(diff -Naupr --ignore-matching-lines='^\s*\"GoVersion\":' --ignore-matching-line='^\s*\"GodepVersion\":' --ignore-matching-lines='^\s*\"Comment\":' ${GODEPS_FILE} ${_kubetmp}/${GODEPS_FILE})"; then
      if [ "$BASEDIR" = "." ]; then
        PATCH_FILE=godepdiff.patch
      else
        PATCH_FILE=godepdiff_`echo $BASEDIR | sed 's:/:_:'`.patch
      fi

      echo "Your ${GODEPS_FILE} is different:" >&2
      echo "${_out}" >&2
      echo "Deps Verify failed." >&2
      echo "${_out}" > "$PATCH_FILE"
      echo "If you're seeing this locally, run the below command to fix your Godeps.json:" >&2
      echo "patch -p0 < $PATCH_FILE" >&2
      echo "(The above output can be saved as $PATCH_FILE if you're not running this locally)" >&2
      echo "(The patch file should also be exported as a build artifact if run through CI)" >&2
      KEEP_TMP=true
      if [[ -f "$PATCH_FILE" && -d "${ARTIFACTS_DIR:-}" ]]; then
        echo "Copying patch to artifacts.."
        cp "$PATCH_FILE" "${ARTIFACTS_DIR:-}/"
      fi
      ret=1
    fi
  done

  # Test for diffs in vendor
  if ! _out="$(diff -Naupr -x "OWNERS" -x "BUILD" -x "AUTHORS*" -x "CONTRIBUTORS*" vendor ${_kubetmp}/vendor)"; then
    echo "Your vendored results are different:" >&2
    echo "${_out}" >&2
    echo "Deps Verify failed." >&2
    echo "${_out}" > vendordiff.patch
    echo "If you're seeing this locally, run the below command to fix your directories:" >&2
    echo "patch -p0 < vendordiff.patch" >&2
    echo "(The above output can be saved as vendordiff.patch if you're not running this locally)" >&2
    echo "(The patch file should also be exported as a build artifact if run through CI)" >&2
    KEEP_TMP=true
    if [[ -f vendordiff.patch && -d "${ARTIFACTS_DIR:-}" ]]; then
      echo "Copying patch to artifacts.."
      cp vendordiff.patch "${ARTIFACTS_DIR:-}/"
    fi
    ret=1
  fi
popd > /dev/null 2>&1

if [[ ${ret} -gt 0 ]]; then
  exit ${ret}
fi

echo "Deps Verified."
# ex: ts=2 sw=2 et filetype=sh
