#!/bin/bash

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

export KUBERNETES_PROVIDER="kubemark"
export KUBE_CONFIG_FILE="config-default.sh"

REPORT_DIR="${1:-_artifacts}"
KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..

source "${KUBE_ROOT}/cluster/kubemark/util.sh"

detect-master

echo "Dumping logs for kubemark master: ${MASTER_NAME}"
DUMP_ONLY_MASTER_LOGS=true ${KUBE_ROOT}/cluster/log-dump.sh "${REPORT_DIR}"
