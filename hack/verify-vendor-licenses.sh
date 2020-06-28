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

# This script checks whether updating of licenses files is needed
# or not. We should run `hack/update-vendor-licenses.sh` and commit the results,
# if actually updates them.
# Usage: `hack/verify-vendor-licenses.sh`.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

# create a nice clean place to put our new licenses
# must be in the user dir (e.g. KUBE_ROOT) in order for the docker volume mount
# to work with docker-machine on macs
mkdir -p "${KUBE_ROOT}/_tmp"
_tmpdir="$(mktemp -d "${KUBE_ROOT}/_tmp/kube-licenses.XXXXXX")"
#echo "Created workspace: ${_tmpdir}"
function cleanup {
  #echo "Removing workspace: ${_tmpdir}"
  rm -rf "${_tmpdir}"
}
kube::util::trap_add cleanup EXIT

# symlink all vendor subfolders in temp vendor
mkdir -p "${_tmpdir}/vendor"
for child in "${KUBE_ROOT}/vendor"/*
do
  ln -s "${child}" "${_tmpdir}/vendor"
done

ln -s "${KUBE_ROOT}/LICENSE" "${_tmpdir}"
ln -s "${KUBE_ROOT}/staging" "${_tmpdir}"

# Update licenses
LICENSE_ROOT="${_tmpdir}" "${KUBE_ROOT}/hack/update-vendor-licenses.sh"

# Compare licenses
if ! _out="$(diff -Naupr -x OWNERS "${KUBE_ROOT}/LICENSES" "${_tmpdir}/LICENSES")"; then
  echo "Your LICENSES tree is out of date. Run hack/update-vendor-licenses.sh and commit the results." >&2
  echo "${_out}" >&2
  exit 1
fi
