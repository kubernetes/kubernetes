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

set -o errexit
set -o nounset
set -o pipefail

# Make sure the current directory is inside a GOPATH element.
parts="$(echo "${GOPATH:-}" | sed 's/:/ /g')"
subdir="src/k8s.io/kubernetes"
for p in ${parts}; do
    if [[ "$(pwd)/" =~ ^"${p}/${subdir}/" ]]; then
        exit 0
    fi
done
echo "Kubernetes must be built from within a valid GOPATH"
echo "  current dir: $(pwd)"
if [ -z "${GOPATH:-}" ]; then
    echo "  GOPATH is not set"
else
    for p in ${parts}; do
        echo "  looking in:  ${p}/${subdir}"
    done
fi
exit 1
