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

set -e

source $(dirname $0)/common.sh

ETCD_DIR="${KUBE_REPO_ROOT}/_output/etcd"
mkdir -p "${ETCD_DIR}"

echo "+++ Running integration test"

etcd -name test -data-dir ${ETCD_DIR} > "${KUBE_REPO_ROOT}/_output/etcd.log" &
ETCD_PID=$!

sleep 5

${KUBE_TARGET}/linux/amd64/integration

kill $ETCD_PID
