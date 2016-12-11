#!/bin/bash

# Copyright 2014 The Kubernetes Authors.
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

# This command checks that the built commands can function together for
# simple scenarios.  It does not require Docker.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
source "${KUBE_ROOT}/hack/make-rules/test-cmd-util.sh"

function run_kube_apiserver() {
  kube::log::status "Building kube-apiserver"
  make -C "${KUBE_ROOT}" WHAT="cmd/kube-apiserver"

  # Start kube-apiserver
  kube::log::status "Starting kube-apiserver"

  # Admission Controllers to invoke prior to persisting objects in cluster
  ADMISSION_CONTROL="NamespaceLifecycle,LimitRanger,ResourceQuota"

  "${KUBE_OUTPUT_HOSTBIN}/kube-apiserver" \
    --address="127.0.0.1" \
    --public-address-override="127.0.0.1" \
    --port="${API_PORT}" \
    --admission-control="${ADMISSION_CONTROL}" \
    --etcd-servers="http://${ETCD_HOST}:${ETCD_PORT}" \
    --public-address-override="127.0.0.1" \
    --kubelet-port=${KUBELET_PORT} \
    --runtime-config=api/v1 \
    --storage-media-type="${KUBE_TEST_API_STORAGE_TYPE-}" \
    --cert-dir="${TMPDIR:-/tmp/}" \
    --service-cluster-ip-range="10.0.0.0/24" 1>&2 &
  APISERVER_PID=$!

  kube::util::wait_for_url "http://127.0.0.1:${API_PORT}/healthz" "apiserver"
}

function run_kube_controller_manager() {
  kube::log::status "Building kube-controller-manager"
  make -C "${KUBE_ROOT}" WHAT="cmd/kube-controller-manager"

  # Start controller manager
  kube::log::status "Starting controller-manager"
  "${KUBE_OUTPUT_HOSTBIN}/kube-controller-manager" \
    --port="${CTLRMGR_PORT}" \
    --kube-api-content-type="${KUBE_TEST_API_TYPE-}" \
    --master="127.0.0.1:${API_PORT}" 1>&2 &
  CTLRMGR_PID=$!

  kube::util::wait_for_url "http://127.0.0.1:${CTLRMGR_PORT}/healthz" "controller-manager"
}

function run_kubelet() {
  # Only run kubelet on platforms it supports
  if [[ "$(go env GOHOSTOS)" == "linux" ]]; then
    kube::log::status "Building kubelet"
    make -C "${KUBE_ROOT}" WHAT="cmd/kubelet"

    kube::log::status "Starting kubelet in masterless mode"
    "${KUBE_OUTPUT_HOSTBIN}/kubelet" \
      --really-crash-for-testing=true \
      --root-dir=/tmp/kubelet.$$ \
      --cert-dir="${TMPDIR:-/tmp/}" \
      --docker-endpoint="fake://" \
      --hostname-override="127.0.0.1" \
      --address="127.0.0.1" \
      --port="$KUBELET_PORT" \
      --healthz-port="${KUBELET_HEALTHZ_PORT}" 1>&2 &
    KUBELET_PID=$!
    kube::util::wait_for_url "http://127.0.0.1:${KUBELET_HEALTHZ_PORT}/healthz" "kubelet(masterless)"
    kill ${KUBELET_PID} 1>&2 2>/dev/null

    kube::log::status "Starting kubelet in masterful mode"
    "${KUBE_OUTPUT_HOSTBIN}/kubelet" \
      --really-crash-for-testing=true \
      --root-dir=/tmp/kubelet.$$ \
      --cert-dir="${TMPDIR:-/tmp/}" \
      --docker-endpoint="fake://" \
      --hostname-override="127.0.0.1" \
      --address="127.0.0.1" \
      --api-servers="${API_HOST}:${API_PORT}" \
      --port="$KUBELET_PORT" \
      --healthz-port="${KUBELET_HEALTHZ_PORT}" 1>&2 &
    KUBELET_PID=$!

    kube::util::wait_for_url "http://127.0.0.1:${KUBELET_HEALTHZ_PORT}/healthz" "kubelet"
  fi
}

# Creates a node object with name 127.0.0.1 if it doesnt exist already.
# This is required for non-linux platforms where we do not run kubelet.
function create_node() {
  if [[ "$(go env GOHOSTOS)" == "linux" ]]; then
    kube::util::wait_for_url "http://127.0.0.1:${API_PORT}/api/v1/nodes/127.0.0.1" "apiserver(nodes)"
  else
    # create a fake node
    kubectl create -f - -s "http://127.0.0.1:${API_PORT}" << __EOF__
{
  "kind": "Node",
  "apiVersion": "v1",
  "metadata": {
    "name": "127.0.0.1"
  },
  "status": {
    "capacity": {
      "memory": "1Gi"
    }
  }
}
__EOF__
  fi
}

kube::log::status "Running kubectl tests for kube-apiserver"

setup
run_kube_apiserver
run_kube_controller_manager
run_kubelet
create_node
SUPPORTED_RESOURCES=("*")
output_message=$(runTests "SUPPORTED_RESOURCES=${SUPPORTED_RESOURCES[@]}")
# Ensure that tests were run. We cannot check all resources here. We check a few
# to catch bugs due to which no tests run.
kube::test::if_has_string "${output_message}" "Testing kubectl(v1:pods)"
kube::test::if_has_string "${output_message}" "Testing kubectl(v1:services)"

kube::log::status "TESTS PASSED"
