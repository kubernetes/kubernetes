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

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
. "${KUBE_ROOT}/cluster/kube-util.sh"

function detect-master () {
  echo "Running locally"
  KUBE_MASTER=127.0.0.1
  KUBE_MASTER_IP=127.0.0.1
}

function test-build-release {
  # Make a release
  "${KUBE_ROOT}/build/release.sh"
}

function kube-up {
  ("${KUBE_ROOT}/hack/local-up-cluster.sh" &)
}

function kube-down {
  ps -ef | grep local-up-cluster.sh | awk '{print $2}' | xargs kill
}

function prepare-e2e() {
  echo "Running e2e locally"
}
