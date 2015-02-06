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

# Launches a container and verifies it can be reached. Assumes that
# we're being called by hack/e2e-test.sh (we use some env vars it sets up).

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

prepare-e2e

liveness_tests="http exec"
if [[ ${KUBERNETES_PROVIDER} == "gke" ]]; then
  server_version=$(kube_server_version)
  if [[ ${server_version} -le 702 ]]; then
    echo "GKE server version <= 0.7.2, limiting test to http (version = ${server_version})"
    liveness_tests="http"
  fi
fi

function teardown() {
  echo "Cleaning up test artifacts"
  for test in ${liveness_tests}; do
    ${KUBECTL} delete pods liveness-${test}
  done
}

function waitForNotPending() {
  pod_id_list=$(${KUBECTL} get pods -o template '--template={{range.items}}{{.id}} {{end}}' -l test=liveness)
  # Pod turn up on a clean cluster can take a while for the docker image pull.
  all_running=0
  for i in $(seq 1 24); do
    echo "Waiting for pod to come up."
    sleep 5
    all_running=1
    for id in $pod_id_list; do
      current_status=$(${KUBECTL} get pods $id -o template '--template={{.currentState.status}}') || true
      if [[ "$current_status" == "Pending" ]]; then
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

trap "teardown" EXIT

for test in ${liveness_tests}; do
  echo "Liveness test: ${test}"
  ${KUBECTL} create -f ${KUBE_ROOT}/examples/liveness/${test}-liveness.yaml
  waitForNotPending

  before=$(${KUBECTL} get pods "liveness-${test}" -o template '--template={{.currentState.info.liveness.restartCount}}')
  while [[ "${before}" == "<no value>" ]]; do
    before=$(${KUBECTL} get pods "liveness-${test}" -o template '--template={{.currentState.info.liveness.restartCount}}')
  done

  echo "Waiting for restarts."
  for i in $(seq 1 24); do
    sleep 10
    after=$(${KUBECTL} get pods "liveness-${test}" -o template '--template={{.currentState.info.liveness.restartCount}}')
    echo "Restarts: ${after} > ${before}"
    if [[ "${after}" == "<no value>" ]]; then
      continue
    fi
    if [[ "${after}" > "${before}" ]]; then
      break
    fi
  done

  if [[ "${before}" < "${after}" ]]; then
    continue
  fi

  echo "Unexpected absence of failures in ${test}"
  echo "Restarts before: ${before}."
  echo "Restarts after: ${after}"
  exit 1
done

exit 0
