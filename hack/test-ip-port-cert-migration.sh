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

# Script to test cluster/update-storage-objects.sh works as expected.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"
source "${KUBE_ROOT}/hack/lib/test.sh"
source "${KUBE_ROOT}/test/cmd/legacy-script.sh"

ETCD_HOST=${ETCD_HOST:-127.0.0.1}
ETCD_PORT=${ETCD_PORT:-2379}
ETCD_PREFIX=${ETCD_PREFIX:-randomPrefix}
API_PORT=${API_PORT:-8080}
API_HOST=${API_HOST:-127.0.0.1}
OLD_PORT=${OLD_PORT:-10443}
NEW_PORT=${NEW_PORT:-9443}
KUBE_API_VERSIONS=""
RUNTIME_CONFIG=""
CERT_DIR=${CERT_DIR:-"${KUBE_ROOT}/hack/ip-cert-test-data"}
TMP_DIR=${TMP_DIR:-"/tmp"}

KUBECTL="${KUBE_OUTPUT_HOSTBIN}/kubectl"

function startDoublePortApiServer() {
  ls -l "${KUBE_OUTPUT_HOSTBIN}/kube-apiserver"

  "${KUBE_OUTPUT_HOSTBIN}/kube-apiserver" \
    --insecure-bind-address="${API_HOST}" \
    --bind-address="${API_HOST}" \
    --insecure-port="${API_PORT}" \
    --secure-port="${NEW_PORT}" \
    --etcd-servers="http://${ETCD_HOST}:${ETCD_PORT}" \
    --etcd-prefix="/${ETCD_PREFIX}" \
    --runtime-config="${RUNTIME_CONFIG}" \
    --cert-dir="${CERT_DIR}" \
    --service-cluster-ip-range="10.0.0.0/24" \
    --old-tls-port="${OLD_PORT}" \
    --tls-cert-file="${CERT_DIR}/oldapiserver.crt" \
    --tls-private-key-file="${CERT_DIR}/oldapiserver.key" \
    --new-tls-cert-file="${CERT_DIR}/apiserver.crt" \
    --new-tls-private-key-file="${CERT_DIR}/apiserver.key" \
    --client-ca-file="${CERT_DIR}/ca-certificates.crt" \
    --service-account-key-file="${CERT_DIR}/serviceaccount.crt" \
    --kubelet-client-certificate="${CERT_DIR}/apiserver-client.crt" \
    --kubelet-client-key="${CERT_DIR}/apiserver-client.key" \
     &
  APISERVER_PID=$!

  # url, prefix, wait, times
  kube::util::wait_for_url "http://${API_HOST}:${API_PORT}/healthz" "apiserver: " 1 120
}

function killApiServer() {
  kube::log::status "Killing api server"
  if [[ -n ${APISERVER_PID-} ]]; then
    kill ${APISERVER_PID} 1>&2 2>/dev/null
    wait ${APISERVER_PID} || true
    kube::log::status "api server exited"
  fi
  unset APISERVER_PID
}

function cleanup() {
  killApiServer

  kube::etcd::cleanup

  kube::log::status "Clean up complete"
}

trap cleanup EXIT SIGINT

make -C "${KUBE_ROOT}" WHAT=cmd/kube-apiserver

kube::etcd::start

# Run on both ports
startDoublePortApiServer

# Make sure both ports work
echo | openssl s_client -showcerts -connect "${API_HOST}:${OLD_PORT}" > "${TMP_DIR}/old.read.crt"
echo | openssl s_client -showcerts -connect "${API_HOST}:${NEW_PORT}" > "${TMP_DIR}/new.read.crt"

killApiServer

# Make sure the certs returned match the certs we passed in flags.
if [ "$(diff --new-line-format='%L' --old-line-format="" --unchanged-line-format="" "${TMP_DIR}/old.read.crt" "${CERT_DIR}/oldapiserver.crt")" != "" ]; then
  echo "Old cert didn\'t match"
  exit 1
fi

if [ "$(diff --new-line-format='%L' --old-line-format="" --unchanged-line-format="" "${TMP_DIR}/new.read.crt" "${CERT_DIR}/apiserver.crt")" != "" ]; then
  echo "New cert didn\'t match"
  exit 1
fi

# Make sure client CAs didn't cross-pollinate
if [ -n "$(grep 0a10eb3f-7265-4679-ab1a-e0f040e345fa "${TMP_DIR}/old.read.crt")" ]; then
  echo "Old cert mentions new client ca!"
  exit 1
fi

if [ -n "$(grep c15f4765-ace3-4179-b352-bb1a4a59f1a6 "${TMP_DIR}/new.read.crt")" ]; then
  echo "New cert mentions old client ca!"
  exit 1
fi
