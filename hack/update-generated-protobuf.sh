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

# This script generates all go files from the corresponding protobuf files.
# Usage: `hack/update-generated-protobuf.sh`.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..

# NOTE: All output from this script needs to be copied back to the calling
# source tree.  This is managed in kube::build::copy_output in build/common.sh.
# If the output set is changed update that function.

APIROOTS=${APIROOTS:-$( \
    git grep --untracked --null -l \
        -e '// +k8s:protobuf-gen=package' \
        -- \
        cmd pkg staging \
        | xargs -0 -n1 dirname \
        | sed 's,^,k8s.io/kubernetes/,;s,k8s.io/kubernetes/staging/src/,,' \
        | sort -u
)}

function git_find() {
    # Similar to find but faster and easier to understand.  We want to include
    # modified and untracked files because this might be running against code
    # which is not tracked by git yet.
    git ls-files -cmo --exclude-standard ':!:vendor/*' "$@"
}

git_find -z ':(glob)**/generated.proto' | xargs -0 rm -f
git_find -z ':(glob)**/generated.pb.go' | xargs -0 rm -f

"${KUBE_ROOT}/build/run.sh" hack/update-generated-protobuf-dockerized.sh "${APIROOTS}" "$@"
