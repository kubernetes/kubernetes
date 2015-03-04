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

: ${KUBE_VERSION_ROOT:=${KUBE_ROOT}}
: ${KUBECTL:="${KUBE_VERSION_ROOT}/cluster/kubectl.sh"}
: ${KUBE_CONFIG_FILE:="config-test.sh"}

export KUBECTL KUBE_CONFIG_FILE

source "${KUBE_ROOT}/cluster/kube-env.sh"
source "${KUBE_VERSION_ROOT}/cluster/${KUBERNETES_PROVIDER}/util.sh"

function wait_for_running() {
  echo "Waiting for pods to come up."
  frontends=$(${KUBECTL} get pods -l name=frontend -o template '--template={{range.items}}{{.id}} {{end}}')
  master=$(${KUBECTL} get pods -l name=redis-master -o template '--template={{range.items}}{{.id}} {{end}}')
  slaves=$(${KUBECTL} get pods -l name=redisslave -o template '--template={{range.items}}{{.id}} {{end}}')
  pods=$(echo $frontends $master $slaves)

  all_running=0
  for i in $(seq 1 30); do
    sleep 10
    all_running=1
    for pod in $pods; do
      status=$(${KUBECTL} get pods $pod -o template '--template={{.currentState.status}}') || true
      if [[ "$status" == "Pending" ]]; then
        all_running=0
        break
      fi
    done
    if [[ "${all_running}" == 1 ]]; then
      break
    fi
  done
  if [[ "${all_running}" == 0 ]]; then
    echo "Pods did not come up in time"
    exit 1
  fi
}

function teardown() {
  ${KUBECTL} stop -f "${GUESTBOOK}" 
}

prepare-e2e

GUESTBOOK="${KUBE_ROOT}/examples/guestbook"

# Launch the guestbook example
${KUBECTL} create -f "${GUESTBOOK}"

trap "teardown" EXIT

# Verify that all pods are running
wait_for_running

get-password
detect-master

# Add a new entry to the guestbook and verify that it was remembered 
FRONTEND_ADDR=https://${KUBE_MASTER_IP}/api/v1beta1/proxy/services/frontend
echo "Waiting for frontend to serve content"
serving=0
for i in $(seq 1 12); do
  ENTRY=$(curl ${FRONTEND_ADDR}/index.php?cmd=get\&key=messages --insecure --user ${KUBE_USER}:${KUBE_PASSWORD})
  echo $ENTRY
  if [[ $ENTRY == '{"data": ""}' ]]; then
    serving=1
    break
  fi
  sleep 10
done
if [[ "${serving}" == 0 ]]; then
  echo "Pods did not start serving content in time"
  exit 1
fi

curl ${FRONTEND_ADDR}/index.php?cmd=set\&key=messages\&value=TestEntry --insecure --user ${KUBE_USER}:${KUBE_PASSWORD}
ENTRY=$(curl ${FRONTEND_ADDR}/index.php?cmd=get\&key=messages --insecure --user ${KUBE_USER}:${KUBE_PASSWORD})

if [[ $ENTRY != '{"data": "TestEntry"}' ]]; then
  echo "Wrong entry received: ${ENTRY}"
  exit 1
fi

exit 0
