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

set -e

HAVE_JQ=$(which jq)
if [[ -z ${HAVE_JQ} ]]; then
  echo "Please install jq, e.g.: 'sudo apt-get install jq' or, "
  echo "if you're on a mac with homebrew, 'brew install jq'."
  exit 1
fi

source "${KUBE_REPO_ROOT}/cluster/util.sh"
GUESTBOOK="${KUBE_REPO_ROOT}/examples/guestbook"

# Launch the guestbook example
$CLOUDCFG -c "${GUESTBOOK}/redis-master.json" create /pods
$CLOUDCFG -c "${GUESTBOOK}/redis-master-service.json" create /services
$CLOUDCFG -c "${GUESTBOOK}/redis-slave-controller.json" create /replicationControllers

sleep 5

POD_LIST_1=$($CLOUDCFG -json list pods | jq ".items[].id")
echo "Pods running: ${POD_LIST_1}"

$CLOUDCFG stop redisSlaveController
# Needed until issue #103 gets fixed
sleep 25
$CLOUDCFG rm redisSlaveController
$CLOUDCFG delete services/redismaster
$CLOUDCFG delete pods/redis-master-2

POD_LIST_2=$($CLOUDCFG -json list pods | jq ".items[].id")
echo "Pods running after shutdown: ${POD_LIST_2}"

exit 0
