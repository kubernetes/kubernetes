#!/usr/bin/env bash

# Copyright The Kubernetes Authors.
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

# This scripts ensures that all items are handled after we cut a new release.
# Usage: `hack/verify-release.sh`.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

res=0

kube::version::get_version_vars
baseVersion=$(sed -n 's/.*DefaultKubeBinaryVersion = "\([^"]*\)".*/\1/p' "${KUBE_ROOT}/staging/src/k8s.io/component-base/version/base.go")

# make sure to remove the '+' sign from KUBE_GIT_MINOR
gitMinor=$(( ${KUBE_GIT_MINOR//+/} ))
gitVersion="${KUBE_GIT_MAJOR}.${gitMinor}"
echo -n "Verifying DefaultKubeBinaryVersion... "
if [[ "${baseVersion}" != "${gitVersion}" ]]; then
  echo "FAILED!"
  echo
  echo "Update DefaultKubeBinaryVersion in ${KUBE_ROOT}/staging/src/k8s.io/component-base/version/base.go"
  echo "See commits in:"
  echo "https://github.com/kubernetes/kubernetes/commits/master/staging/src/k8s.io/component-base/version/base.go"
  echo
  res=1
else
  echo "OK"
fi

# We need to ensure new testdata for the previous version exists
previousMinor=$((gitMinor - 1))
newTestDataDir="${KUBE_ROOT}/staging/src/k8s.io/api/testdata/v${KUBE_GIT_MAJOR}.${previousMinor}.0"
# We need to ensure old (current version - 4) testdata is removed,
# we keep 3 previous testdata versions.
oldMinor=$((gitMinor - 4))
oldTestDataDir="${KUBE_ROOT}/staging/src/k8s.io/api/testdata/v${KUBE_GIT_MAJOR}.${oldMinor}.0"
echo -n "Verifying testdata... "
if [[ ! -d "${newTestDataDir}" || -d "${oldTestDataDir}" ]]; then
  echo "FAILED!"
  echo
  if [ ! -d "${newTestDataDir}" ]; then
    echo "Missing testdata in ${newTestDataDir}"
  fi
  if [ -d "${oldTestDataDir}" ]; then
    echo "Unnecessary testdata in ${oldTestDataDir}"
  fi
  echo "See: https://github.com/kubernetes/kubernetes/tree/master/staging/src/k8s.io/api/testdata/README.md"
  echo "and commits in:"
  echo "https://github.com/kubernetes/kubernetes/commits/master/staging/src/k8s.io/api/testdata"
  echo
  res=1
else
  echo "OK"
fi

exit "$res"
# ex: ts=2 sw=2 et filetype=sh
