#!/usr/bin/env bash

# Copyright 2021 The Kubernetes Authors.
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

# This script replace "net" stdlib IP and CIDR parsers
# with the ones forked in k8s.io/utils/net to parse IP addresses
# because of the compatibility break introduced in golang 1.17
# Reference: #100895
# Usage: `hack/update-netparse-cve.sh`.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"
source "${KUBE_ROOT}/hack/lib/util.sh"

kube::golang::setup_env

# Ensure that we find the binaries we build before anything else.
export GOBIN="${KUBE_OUTPUT_BIN}"
PATH="${GOBIN}:${PATH}"

# Install golangci-lint
echo 'installing net parser converter'
go -C "${KUBE_ROOT}/hack/tools" install github.com/aojea/sloppy-netparser

cd "${KUBE_ROOT}"

function git_find() {
    # Similar to find but faster and easier to understand.  We want to include
    # modified and untracked files because this might be running against code
    # which is not tracked by git yet.
    git ls-files -cmo --exclude-standard \
        ':!:vendor/*'        `# catches vendor/...` \
        ':!:*/vendor/*'      `# catches any subdir/vendor/...` \
        ':!:third_party/*'   `# catches third_party/...` \
        ':!:*/third_party/*' `# catches any subdir/third_party/...` \
        ':!:*/testdata/*'    `# catches any subdir/testdata/...` \
        ':(glob)**/*.go' \
        "$@"
}

# replace net.ParseIP() and netParseIPCDR
git_find -z | xargs -0 sloppy-netparser
