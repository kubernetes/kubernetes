#!/bin/bash

# Copyright 2015 Google Inc. All rights reserved.
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

# Clean out the output directory on the docker host.
set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../../

version=${1:-""}
if [[ "${version}" == "" ]]; then
    echo "Usage: build.sh <version>"
    exit 1
fi

binaries="kube-apiserver kube-scheduler kube-controller-manager"

for x in ${binaries}; do
 cp "${KUBE_ROOT}/_output/release-stage/server/linux-amd64/kubernetes/server/bin/${x}" "${x}"
done

docker build -t "kubernetes/kube-master:${version}" .

for x in ${binaries}; do
  rm $x
done
