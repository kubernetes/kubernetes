#!/usr/bin/env bash

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

# Script to fetch latest openapi spec.
# Puts the updated spec at api/openapi-spec/

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
OPENAPI_ROOT_DIR="${KUBE_ROOT}/api/openapi-spec"
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::util::require-jq
kube::golang::setup_env

make -C "${KUBE_ROOT}" WHAT=cmd/kube-apiserver

function cleanup()
{
    if [[ -n ${APISERVER_PID-} ]]; then
      kill "${APISERVER_PID}" 1>&2 2>/dev/null
      wait "${APISERVER_PID}" || true
    fi
    unset APISERVER_PID

    kube::etcd::cleanup

    kube::log::status "Clean up complete"
}

trap cleanup EXIT SIGINT

kube::golang::setup_env

TMP_DIR=$(mktemp -d /tmp/update-openapi-spec.XXXX)
ETCD_HOST=${ETCD_HOST:-127.0.0.1}
ETCD_PORT=${ETCD_PORT:-2379}
API_PORT=${API_PORT:-8050}
API_HOST=${API_HOST:-127.0.0.1}
API_LOGFILE=${API_LOGFILE:-/tmp/openapi-api-server.log}

kube::etcd::start

echo "dummy_token,admin,admin" > "${TMP_DIR}/tokenauth.csv"

# setup envs for TokenRequest required flags
SERVICE_ACCOUNT_LOOKUP=${SERVICE_ACCOUNT_LOOKUP:-true}
SERVICE_ACCOUNT_KEY=${SERVICE_ACCOUNT_KEY:-/tmp/kube-serviceaccount.key}
# Generate ServiceAccount key if needed
if [[ ! -f "${SERVICE_ACCOUNT_KEY}" ]]; then
  mkdir -p "$(dirname "${SERVICE_ACCOUNT_KEY}")"
  openssl genrsa -out "${SERVICE_ACCOUNT_KEY}" 2048 2>/dev/null
fi

# Start kube-apiserver
kube::log::status "Starting kube-apiserver"
"${KUBE_OUTPUT_HOSTBIN}/kube-apiserver" \
  --bind-address="${API_HOST}" \
  --secure-port="${API_PORT}" \
  --etcd-servers="http://${ETCD_HOST}:${ETCD_PORT}" \
  --advertise-address="10.10.10.10" \
  --cert-dir="${TMP_DIR}/certs" \
  --runtime-config="api/all=true" \
  --token-auth-file="${TMP_DIR}/tokenauth.csv" \
  --authorization-mode=RBAC \
  --service-account-key-file="${SERVICE_ACCOUNT_KEY}" \
  --service-account-lookup="${SERVICE_ACCOUNT_LOOKUP}" \
  --service-account-issuer="https://kubernetes.default.svc" \
  --service-account-signing-key-file="${SERVICE_ACCOUNT_KEY}" \
  --logtostderr \
  --v=2 \
  --service-cluster-ip-range="10.0.0.0/24" >"${API_LOGFILE}" 2>&1 &
APISERVER_PID=$!

if ! kube::util::wait_for_url "https://${API_HOST}:${API_PORT}/healthz" "apiserver: "; then
  kube::log::error "Here are the last 10 lines from kube-apiserver (${API_LOGFILE})"
  kube::log::error "=== BEGIN OF LOG ==="
  tail -10 "${API_LOGFILE}" >&2 || :
  kube::log::error "=== END OF LOG ==="
  exit 1
fi

kube::log::status "Updating " "${OPENAPI_ROOT_DIR}"

curl -w "\n" -kfsS -H 'Authorization: Bearer dummy_token' "https://${API_HOST}:${API_PORT}/openapi/v2" | jq -S '.info.version="unversioned"' > "${OPENAPI_ROOT_DIR}/swagger.json"

kube::log::status "SUCCESS"

# ex: ts=2 sw=2 et filetype=sh
