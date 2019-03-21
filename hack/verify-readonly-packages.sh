#!/usr/bin/env bash

# Copyright 2017 The Kubernetes Authors.
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

# verify-readonly-packages.sh checks whether between $KUBE_VERIFY_GIT_BRANCH and HEAD files
# in readonly directories were modified. A directory is readonly iff it contains a .readonly
# file. Being readonly DOES NOT apply recursively to subdirectories.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

readonly branch=${1:-${KUBE_VERIFY_GIT_BRANCH:-master}}

find_files() {
  find . -not \( \
      \( \
        -wholename './output' \
        -o -wholename './_output' \
        -o -wholename './_gopath' \
        -o -wholename './release' \
        -o -wholename './target' \
        -o -wholename '*/third_party/*' \
        -o -wholename '*/vendor/*' \
        -o -wholename './staging/src/k8s.io/client-go/*vendor/*' \
        -o -wholename './staging/src/k8s.io/client-go/pkg/*' \
      \) -prune \
    \) -name '.readonly'
}

IFS=$'\n'
conflicts=($(find_files | sed 's|/.readonly||' | while read dir; do
    dir=${dir#./}
    if kube::util::has_changes "${branch}" "^${dir}/[^/]*\$" '/\.readonly$|/BUILD$|/zz_generated|/\.generated\.|\.proto$|\.pb\.go$' >/dev/null; then
        echo "${dir}"
    fi
done))
unset IFS

if [ ${#conflicts[@]} -gt 0 ]; then
    exec 1>&2
    for dir in "${conflicts[@]}"; do
        echo "Found ${dir}/.readonly, but files changed compared to \"${branch}\" branch."
    done
    exit 1
else
    echo "Readonly packages verified."
fi
# ex: ts=2 sw=2 et filetype=sh
