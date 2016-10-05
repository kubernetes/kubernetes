#!/bin/bash

# Copyright 2016 The Kubernetes Authors.
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

# This script sets up a temporary Kubernetes GOPATH and runs an arbitrary
# command under it.  Go tooling requires that the current directory be under
# GOPATH or else it fails to find some things, such as the vendor directory for
# the project.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

CWD=$(pwd -P)
IN_GOPATH=false
for gp in ${GOPATH//:/ }; do
    if [[ "${CWD#${gp}}" != "${CWD}" ]]; then
        IN_GOPATH=true
        break
    fi
done
if [[ "${IN_GOPATH}" != "true" ]]; then
    echo "Current directory must be under GOPATH" >/dev/stderr
    exit 1
fi

exit 0
