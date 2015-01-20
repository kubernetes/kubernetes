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

# Assumes a running Kubernetes test cluster; verifies that the guestbook example
# works. Assumes that we're being called by hack/e2e-test.sh (we use some env
# vars it sets up).

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
source "${KUBE_ROOT}/cluster/kube-env.sh"
source "${KUBE_ROOT}/cluster/$KUBERNETES_PROVIDER/util.sh"

GUESTBOOK="${KUBE_ROOT}/examples/guestbook"

# Launch the guestbook example
$KUBECFG -c "${GUESTBOOK}/redis-master.json" create /pods
$KUBECFG -c "${GUESTBOOK}/redis-master-service.json" create /services
$KUBECFG -c "${GUESTBOOK}/redis-slave-controller.json" create /replicationControllers

sleep 5

POD_LIST_1=$($KUBECFG '-template={{range.items}}{{.id}} {{end}}' list pods)
echo "Pods running: ${POD_LIST_1}"

$KUBECFG stop redis-slave-controller
# Needed until issue #103 gets fixed
sleep 25
$KUBECFG rm redis-slave-controller
$KUBECFG delete services/redis-master
$KUBECFG delete pods/redis-master

POD_LIST_2=$($KUBECFG '-template={{range.items}}{{.id}} {{end}}' list pods)
echo "Pods running after shutdown: ${POD_LIST_2}"

exit 0
