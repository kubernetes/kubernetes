#!/usr/bin/env bash
# Copyright 2018 The Kubernetes Authors.
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

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"
source "${KUBE_ROOT}/hack/lib/util.sh"

kube::golang::verify_go_version

# Ensure that we find the binaries we build before anything else.
export GOBIN="${KUBE_OUTPUT_BINPATH}"
PATH="${GOBIN}:${PATH}"

# Install tools we need, but only from vendor/...
go install k8s.io/kubernetes/vendor/github.com/golangci/golangci-lint/cmd/golangci-lint

# golangci-line check
# All the skipping files are defined in hack/.spelling_failures
for path in $(find ${KUBE_ROOT}/cmd -maxdepth 1 -type d) ;do
    if [ ${path} = ${KUBE_ROOT}/cmd ] ;then
        continue
    fi
    golangci-lint run -v ${path} -c ${KUBE_ROOT}/hack/.golangci.yml
done

