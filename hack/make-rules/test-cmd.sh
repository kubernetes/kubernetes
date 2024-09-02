#!/usr/bin/env bash

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

# start the cache mutation detector by default so that cache mutators will be found
KUBE_CACHE_MUTATION_DETECTOR="${KUBE_CACHE_MUTATION_DETECTOR:-true}"
export KUBE_CACHE_MUTATION_DETECTOR

# panic the server on watch decode errors since they are considered coder mistakes
KUBE_PANIC_WATCH_DECODE_ERROR="${KUBE_PANIC_WATCH_DECODE_ERROR:-true}"
export KUBE_PANIC_WATCH_DECODE_ERROR

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/../..
source "${KUBE_ROOT}/hack/lib/init.sh"
source "${KUBE_ROOT}/hack/lib/test.sh"
source "${KUBE_ROOT}/test/cmd/legacy-script.sh"

# setup envs for TokenRequest required flags
SERVICE_ACCOUNT_LOOKUP=${SERVICE_ACCOUNT_LOOKUP:-true}
SERVICE_ACCOUNT_KEY=${SERVICE_ACCOUNT_KEY:-/tmp/kube-serviceaccount.key}
# Generate ServiceAccount key if needed
if [[ ! -f "${SERVICE_ACCOUNT_KEY}" ]]; then
  mkdir -p "$(dirname "${SERVICE_ACCOUNT_KEY}")"
  openssl genrsa -out "${SERVICE_ACCOUNT_KEY}" 2048 2>/dev/null
fi

# Runs kube-apiserver
#
# Exports:
#   APISERVER_PID
function run_kube_apiserver() {
  kube::log::status "Building kube-apiserver"
  make -C "${KUBE_ROOT}" WHAT="cmd/kube-apiserver"

  # Start kube-apiserver
  kube::log::status "Starting kube-apiserver"

  # Admission Controllers to invoke prior to persisting objects in cluster
  ENABLE_ADMISSION_PLUGINS="LimitRanger,ResourceQuota"
  DISABLE_ADMISSION_PLUGINS="ServiceAccount,DefaultStorageClass,DefaultTolerationSeconds,MutatingAdmissionWebhook,ValidatingAdmissionWebhook,StorageObjectInUseProtection"

  # Include RBAC (to exercise bootstrapping), and AlwaysAllow to allow all actions
  AUTHORIZATION_MODE="RBAC,AlwaysAllow"

  # Enable features
  ENABLE_FEATURE_GATES=""

  VERSION_OVERRIDE=""
  if [[ "${CUSTOM_VERSION_SUFFIX:-}" != "" ]]; then
    VERSION_OVERRIDE="--version=$("${THIS_PLATFORM_BIN}/kube-apiserver" --version | awk '{print $2}')${CUSTOM_VERSION_SUFFIX:-}"
  fi

  "${THIS_PLATFORM_BIN}/kube-apiserver" \
    ${VERSION_OVERRIDE:+"${VERSION_OVERRIDE}"} \
    --bind-address="127.0.0.1" \
    --authorization-mode="${AUTHORIZATION_MODE}" \
    --secure-port="${SECURE_API_PORT}" \
    --feature-gates="${ENABLE_FEATURE_GATES}" \
    --enable-admission-plugins="${ENABLE_ADMISSION_PLUGINS}" \
    --disable-admission-plugins="${DISABLE_ADMISSION_PLUGINS}" \
    --etcd-servers="http://${ETCD_HOST}:${ETCD_PORT}" \
    --runtime-config=api/v1 \
    --service-account-key-file="${SERVICE_ACCOUNT_KEY}" \
    --service-account-lookup="${SERVICE_ACCOUNT_LOOKUP}" \
    --service-account-issuer="https://kubernetes.default.svc" \
    --service-account-signing-key-file="${SERVICE_ACCOUNT_KEY}" \
    --storage-media-type="${KUBE_TEST_API_STORAGE_TYPE-}" \
    --cert-dir="${TMPDIR:-/tmp/}" \
    --service-cluster-ip-range="10.0.0.0/24" \
    --client-ca-file=hack/testdata/ca/ca.crt \
    --token-auth-file=hack/testdata/auth-tokens.csv 1>&2 &
  export APISERVER_PID=$!

  kube::util::wait_for_url_with_bearer_token "https://127.0.0.1:${SECURE_API_PORT}/healthz" "admin-token" "apiserver"
}

# Runs run_kube_controller_manager
#
# Exports:
#   CTLRMGR_PID
function run_kube_controller_manager() {
  kube::log::status "Building kube-controller-manager"
  make -C "${KUBE_ROOT}" WHAT="cmd/kube-controller-manager"

  # Start controller manager
  kube::log::status 'Generate kubeconfig for controller-manager'
  local config
  config="$(mktemp controller-manager.kubeconfig.XXXXX)"
  cat <<EOF > "$config"
kind: Config
users:
- name: controller-manager
  user:
    token: admin-token
clusters:
- cluster:
    server: https://127.0.0.1:${SECURE_API_PORT}
    insecure-skip-tls-verify: true
  name: local
contexts:
- context:
    cluster: local
    user: controller-manager
  name: local-context
current-context: local-context
EOF

  kube::log::status "Starting controller-manager"
  "${THIS_PLATFORM_BIN}/kube-controller-manager" \
    --kube-api-content-type="${KUBE_TEST_API_TYPE-}" \
    --cluster-signing-cert-file=hack/testdata/ca/ca.crt \
    --cluster-signing-key-file=hack/testdata/ca/ca.key \
    --kubeconfig="${config}" 1>&2 &
  export CTLRMGR_PID=$!

  kube::util::wait_for_url "https://127.0.0.1:${SECURE_CTLRMGR_PORT}/healthz" "controller-manager"
}

# Creates a node object with name 127.0.0.1. This is required because we do not
# run kubelet.
#
# An arbitrary annotation is needed to ensure field managers are saved on the
# object. Without it, we would be creating an empty object and because status
# and name get wiped, there were be no field managers tracking any fields.
#
# Exports:
#   SUPPORTED_RESOURCES(Array of all resources supported by the apiserver).
function create_node() {
  kubectl create -f - << __EOF__
{
  "kind": "Node",
  "apiVersion": "v1",
  "metadata": {
    "name": "127.0.0.1",
    "annotations": {
      "save-managers": "true"
    }
  },
  "status": {
    "capacity": {
      "memory": "1Gi"
    }
  }
}
__EOF__
}

# Run it if:
# 1) $WHAT is empty
# 2) $WHAT is not empty and kubeadm is part of $WHAT
WHAT=${WHAT:-}
if [[ ${WHAT} == "" || ${WHAT} =~ .*kubeadm.* ]] ; then
  kube::log::status "Running kubeadm tests"

  # build kubeadm
  make all -C "${KUBE_ROOT}" WHAT=cmd/kubeadm
  # unless the user sets KUBEADM_PATH, assume that "make all..." just built it
  export KUBEADM_PATH="${KUBEADM_PATH:=$(kube::realpath "${KUBE_ROOT}")/_output/local/go/bin/kubeadm}"
  # invoke the tests
  make -C "${KUBE_ROOT}" test \
    WHAT=k8s.io/kubernetes/cmd/kubeadm/test/cmd \
    KUBE_TIMEOUT=--timeout=240s \
    KUBE_RACE=""

  # if we ONLY want to run kubeadm, then exit here.
  if [[ ${WHAT} == "kubeadm" ]]; then
    kube::log::status "TESTS PASSED"
    exit 0
  fi
fi

kube::log::status "Running kubectl tests for kube-apiserver"

setup

# Test custom version invocation
CUSTOM_VERSION_SUFFIX=-custom run_kube_apiserver
kube::test::if_has_string "$(kubectl get --raw /version)" "gitVersion.*-custom"
kill "${APISERVER_PID}" 1>&2 2>/dev/null
wait "${APISERVER_PID}" || true
unset APISERVER_PID

run_kube_apiserver
run_kube_controller_manager
create_node
export SUPPORTED_RESOURCES=("*")
# WARNING: Do not wrap this call in a subshell to capture output, e.g. output=$(runTests)
# Doing so will suppress errexit behavior inside runTests
runTests

kube::log::status "TESTS PASSED"
