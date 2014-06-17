#!/bin/bash

# Copyright 2014 Google Inc. All rights reserved.
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

# This and builds all go components.

set -e

source $(dirname $0)/common.sh

readonly BINARIES="
  proxy
  integration
  apiserver
  controller-manager
  kubelet
  cloudcfg
  localkube"

if [[ -n $1 ]]; then
  echo "+++ Building $1"
  go build \
    -o "${KUBE_TARGET}/$1" \
    github.com/GoogleCloudPlatform/kubernetes/cmd/$1
  exit 0
fi

for b in ${BINARIES}; do
  echo "+++ Building $b"
  go build \
    -o "${KUBE_TARGET}/$b" \
    github.com/GoogleCloudPlatform/kubernetes/cmd/$b
done
