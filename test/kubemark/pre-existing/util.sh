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

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../../..

source "${KUBE_ROOT}/test/kubemark/common/util.sh"

# Leave the skeleton definition of execute-cmd-on-master-with-retries
# so only the pre-existing provider functions will target this.
function execute-cmd-on-pre-existing-master-with-retries() {
  IP_WITHOUT_PORT=$(echo "${MASTER_IP}" | cut -f 1 -d ':') || "${MASTER_IP}"

  RETRIES="${2:-1}" run-cmd-with-retries ssh kubernetes@"${IP_WITHOUT_PORT}" $1
}
