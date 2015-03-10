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
  if [[ "${KUBERNETES_PROVIDER}" == "gce" ]]; then
    local REGION=${ZONE%-*}
    gcloud compute forwarding-rules delete -q --region ${REGION} "${INSTANCE_PREFIX}-default-frontend" || true
    gcloud compute target-pools delete -q --region ${REGION} "${INSTANCE_PREFIX}-default-frontend" || true
  fi
}

prepare-e2e

trap "teardown" EXIT

echo "WARNING: this test is a no op that only attempts to launch guestbook resources."
# Launch the guestbook example
${KUBECTL} create -f "${GUESTBOOK}"

sleep 15

POD_LIST_1=$(${KUBECTL} get pods -o template '--template={{range.items}}{{.id}} {{end}}')
echo "Pods running: ${POD_LIST_1}"

# TODO make this an actual test. Open up a firewall and use curl to post and
# read a message via the frontend

${KUBECTL} stop -f "${GUESTBOOK}"

POD_LIST_2=$(${KUBECTL} get pods -o template '--template={{range.items}}{{.id}} {{end}}')
echo "Pods running after shutdown: ${POD_LIST_2}"

exit 0
