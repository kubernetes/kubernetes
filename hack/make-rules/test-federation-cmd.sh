#!/bin/bash

# Copyright 2016 The Kubernetes Authors.
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
source "${KUBE_ROOT}/hack/lib/init.sh"
source "${KUBE_ROOT}/hack/lib/test.sh"
source "${KUBE_ROOT}/hack/make-rules/test-cmd-util.sh"

function run_federation_apiserver() {
  kube::log::status "Building federation-apiserver"
  make -C "${KUBE_ROOT}" WHAT="federation/cmd/federation-apiserver"

  # Start federation-apiserver
  kube::log::status "Starting federation-apiserver"

  # Admission Controllers to invoke prior to persisting objects in cluster
  ADMISSION_CONTROL="NamespaceLifecycle"

  "${KUBE_OUTPUT_HOSTBIN}/federation-apiserver" \
    --insecure-port="${API_PORT}" \
    --secure-port="${SECURE_API_PORT}" \
    --admission-control="${ADMISSION_CONTROL}" \
    --etcd-servers="http://${ETCD_HOST}:${ETCD_PORT}" \
    --storage-media-type="${KUBE_TEST_API_STORAGE_TYPE-}" \
    --cert-dir="${TMPDIR:-/tmp/}" \
    --insecure-allow-any-token 1>&2 &
  APISERVER_PID=$!

  kube::util::wait_for_url "http://127.0.0.1:${API_PORT}/healthz" "apiserver"
}

function run_federation_controller_manager() {
  kube::log::status "Building federation-controller-manager"
  make -C "${KUBE_ROOT}" WHAT="federation/cmd/federation-controller-manager"

  # Create a kubeconfig for federation apiserver.
  local kubeconfig="${KUBE_TEMP}/kubeconfig"
  touch "${kubeconfig}"
  kubectl config set-cluster "apiserver" --server="http://127.0.0.1:${API_PORT}" --insecure-skip-tls-verify=true --kubeconfig="${kubeconfig}"
  kubectl config set-context "context" --cluster="apiserver" --kubeconfig="${kubeconfig}"
  kubectl config use-context "context" --kubeconfig="${kubeconfig}"

  # Start controller manager
  kube::log::status "Starting federation-controller-manager"
  "${KUBE_OUTPUT_HOSTBIN}/federation-controller-manager" \
    --port="${CTLRMGR_PORT}" \
    --kubeconfig="${kubeconfig}" \
    --kube-api-content-type="${KUBE_TEST_API_TYPE-}" \
    --controllers="service-dns=false" \
    --master="127.0.0.1:${API_PORT}" 1>&2 &
  CTLRMGR_PID=$!

  kube::util::wait_for_url "http://127.0.0.1:${CTLRMGR_PORT}/healthz" "controller-manager"
}

kube::log::status "Running kubectl tests for federation-apiserver"

setup
run_federation_apiserver
run_federation_controller_manager
# TODO: Fix for replicasets and deployments.
SUPPORTED_RESOURCES=("configmaps" "daemonsets" "events" "ingress" "namespaces" "services" "secrets")
# Set wait for deletion to true for federation apiserver since resources are
# deleted asynchronously.
# This is a temporary workaround until https://github.com/kubernetes/kubernetes/issues/42594 is fixed.
WAIT_FOR_DELETION="true"
# WARNING: Do not wrap this call in a subshell to capture output, e.g. output=$(runTests)
# Doing so will suppress errexit behavior inside runTests
runTests

kube::log::status "TESTS PASSED"
