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

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

readonly branch=${1:-${KUBE_VERIFY_GIT_BRANCH:-master}}
if ! [[ ${KUBE_FORCE_VERIFY_CHECKS:-} =~ ^[yY]$ ]] && \
  ! kube::util::has_changes_against_upstream_branch "${branch}" 'vendor/'; then
  exit 0
fi

# Create a nice clean place to put our new vendored deps
_tmpdir="$(mktemp -d -t gopath.XXXXXX)"
function cleanup {
  echo "Removing ${_tmpdir}"
  rm -rf "${_tmpdir}"
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
cd "${_kubetmp}"

# Build the govendor tool
go get -u github.com/kardianos/govendor 2>/dev/null
GOVENDOR="${GOPATH}/bin/govendor"

# Fill out that nice clean place with the kube vendored deps
echo "Starting to sync all kubernetes vendored deps. This takes a while"
"${GOVENDOR}" sync
echo "Sync finished"

# Test for diffs
if ! _out="$(diff -Naupr ${KUBE_ROOT}/vendor ${_kubetmp}/vendor)"; then
  echo "Your vendor/ dir is different:"
  echo "${_out}"
  echo "Vendored deps verify failed."
  exit 1
fi

echo "Vendored deps verified."
# ex: ts=2 sw=2 et filetype=sh
