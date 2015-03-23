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

GUESTBOOK="${KUBE_ROOT}/examples/guestbook"

function teardown() {
  ${KUBECTL} stop -f "${GUESTBOOK}"
  if [[ "${KUBERNETES_PROVIDER}" == "gce" ]] || [[ "${KUBERNETES_PROVIDER}" == "gke" ]]; then
    local REGION=${ZONE%-*}
    if [[ "${KUBERNETES_PROVIDER}" == "gce" ]]; then
      gcloud compute forwarding-rules delete -q --project="${PROJECT}" --region ${REGION} "${INSTANCE_PREFIX}-default-frontend" || true
      gcloud compute target-pools delete -q --project="${PROJECT}" --region ${REGION} "${INSTANCE_PREFIX}-default-frontend" || true
    fi
    gcloud compute firewall-rules delete guestbook-e2e-minion-8000 -q --project="${PROJECT}" || true
  fi
}

function wait_for_running() {
  echo "Waiting for pods to come up."
  local frontends master slaves pods all_running status i pod
  frontends=($(${KUBECTL} get pods -l name=frontend -o template '--template={{range.items}}{{.id}} {{end}}'))
  master=($(${KUBECTL} get pods -l name=redis-master -o template '--template={{range.items}}{{.id}} {{end}}'))
  slaves=($(${KUBECTL} get pods -l name=redis-slave -o template '--template={{range.items}}{{.id}} {{end}}'))
  pods=("${frontends[@]}" "${master[@]}" "${slaves[@]}")

  all_running=0
  for i in {1..30}; do
    all_running=1
    for pod in "${pods[@]}"; do
      status=$(${KUBECTL} get pods "${pod}" -o template '--template={{.currentState.status}}') || true
      if [[ "$status" != "Running" ]]; then
        all_running=0
        break
      fi
    done
    if [[ "${all_running}" == 1 ]]; then
      break
    fi
    sleep 10
  done
  if [[ "${all_running}" == 0 ]]; then
    echo "Pods did not come up in time"
    return 1
  fi
}

prepare-e2e

trap "teardown" EXIT

# Launch the guestbook example
${KUBECTL} create -f "${GUESTBOOK}"

# Verify that all pods are running
wait_for_running

if [[ "${KUBERNETES_PROVIDER}" == "gce" ]] || [[ "${KUBERNETES_PROVIDER}" == "gke" ]]; then
  gcloud compute firewall-rules create --project="${PROJECT}" --allow=tcp:8000 --network="${NETWORK}" --target-tags="${MINION_TAG}" guestbook-e2e-minion-8000
fi

# Add a new entry to the guestbook and verify that it was remembered
frontend_addr=$(${KUBECTL} get service frontend -o template '--template={{range .publicIPs}}{{.}}{{end}}:{{.port}}')
echo "Waiting for frontend to serve content"
serving=0
for i in {1..12}; do
  entry=$(curl "http://${frontend_addr}/index.php?cmd=get&key=messages") || true
  echo ${entry}
  if [[ "${entry}" == '{"data": ""}' ]]; then
    serving=1
    break
  fi
  sleep 10
done
if [[ "${serving}" == 0 ]]; then
  echo "Pods did not start serving content in time"
  exit 1
fi

curl "http://${frontend_addr}/index.php?cmd=set&key=messages&value=TestEntry"
entry=$(curl "http://${frontend_addr}/index.php?cmd=get&key=messages")

if [[ "${entry}" != '{"data": "TestEntry"}' ]]; then
  echo "Wrong entry received: ${entry}"
  exit 1
fi

exit 0
