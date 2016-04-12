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

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

readonly branch=${1:-${KUBE_VERIFY_GIT_BRANCH:-master}}
if ! [[ ${KUBE_FORCE_VERIFY_CHECKS:-} =~ ^[yY]$ ]] && \
  ! kube::util::has_changes_against_upstream_branch "${branch}" 'vendor/'; then
  exit 0
fi

# create a nice clean place to put stuff
_tmpdir="$(mktemp -d -t gopath.XXXXXX)"
function cleanup {
  echo "Removing ${_tmpdir}"
  rm -rf "${_tmpdir}"
}
trap cleanup EXIT

export GOPATH="${_tmpdir}"

# build the glide tool
go get -u github.com/Masterminds/glide
GLIDE="${_tmpdir}/bin/glide"

# do a refresh of current deps from upstream repos
LOCK="glide.lock"
cat "${LOCK}" > "${_tmpdir}/${LOCK}"
"${GLIDE}" --no-color update --update-vendored --delete --strip-vendor --strip-vcs

# check that nothing changed
echo "Checking ${LOCK} for changes"
# glide updates the timestamp even if nothing else changed
diff -Naupr --ignore-matching-lines='^updated:' "${_tmpdir}/${LOCK}" "${LOCK}"
cat "${_tmpdir}/${LOCK}" > "${LOCK}"
echo "Checking git status for changes"
STATUS="$(git status --short)"
if [[ -n "${STATUS}" ]]; then
  git status
  exit 1
fi

exit 0

# ex: ts=2 sw=2 et filetype=sh
