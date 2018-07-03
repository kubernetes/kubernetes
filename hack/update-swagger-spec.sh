#!/usr/bin/env bash

# Copyright 2015 The Kubernetes Authors.
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

# Script to fetch latest swagger spec.
# Puts the updated spec at api/swagger-spec/

set -o errexit
set -o nounset
set -o pipefail

cat << __EOF__
Note: This assumes that the 'types_swagger_doc_generated.go' file has been
updated for all API group versions. If you are unsure, please run
hack/update-generated-swagger-docs.sh first.
__EOF__

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
SWAGGER_ROOT_DIR="${KUBE_ROOT}/api/swagger-spec"
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::golang::setup_env

make -C "${KUBE_ROOT}" WHAT=cmd/kube-apiserver

function cleanup()
{
    if [[ -n "${APISERVER_PID-}" ]]; then
      kill "${APISERVER_PID}" &>/dev/null || :
      wait "${APISERVER_PID}" &>/dev/null || :
    fi

    kube::etcd::cleanup

    kube::log::status "Clean up complete"
}

kube::util::trap_add cleanup EXIT

kube::golang::setup_env

apiserver=$(kube::util::find-binary "kube-apiserver")

TMP_DIR=$(mktemp -d /tmp/update-swagger-spec.XXXX)
ETCD_HOST=${ETCD_HOST:-127.0.0.1}
ETCD_PORT=${ETCD_PORT:-2379}
API_PORT=${API_PORT:-8050}
API_HOST=${API_HOST:-127.0.0.1}
API_LOGFILE=${API_LOGFILE:-/tmp/swagger-api-server.log}

kube::etcd::start


# Start kube-apiserver, with alpha api versions on so we can harvest their swagger docs
# Set --runtime-config to all versions in KUBE_AVAILABLE_GROUP_VERSIONS to enable alpha features.
kube::log::status "Starting kube-apiserver"
"${KUBE_OUTPUT_HOSTBIN}/kube-apiserver" \
  --insecure-bind-address="${API_HOST}" \
  --bind-address="${API_HOST}" \
  --insecure-port="${API_PORT}" \
  --etcd-servers="http://${ETCD_HOST}:${ETCD_PORT}" \
  --advertise-address="10.10.10.10" \
  --cert-dir="${TMP_DIR}/certs" \
  --runtime-config=$(echo "${KUBE_AVAILABLE_GROUP_VERSIONS}" | sed -E 's|[[:blank:]]+|,|g') \
  --service-cluster-ip-range="10.0.0.0/24" >"${API_LOGFILE}" 2>&1 &
APISERVER_PID=$!

if ! kube::util::wait_for_url "${API_HOST}:${API_PORT}/healthz" "apiserver: "; then
  kube::log::error "Here are the last 10 lines from kube-apiserver (${API_LOGFILE})"
  kube::log::error "=== BEGIN OF LOG ==="
  tail -10 "${API_LOGFILE}" || :
  kube::log::error "=== END OF LOG ==="
  exit 1
fi

SWAGGER_API_PATH="${API_HOST}:${API_PORT}/swaggerapi/"

kube::log::status "Updating " ${SWAGGER_ROOT_DIR}

SWAGGER_API_PATH="${SWAGGER_API_PATH}" SWAGGER_ROOT_DIR="${SWAGGER_ROOT_DIR}" VERSIONS="${KUBE_AVAILABLE_GROUP_VERSIONS}" KUBE_NONSERVER_GROUP_VERSIONS="${KUBE_NONSERVER_GROUP_VERSIONS}" kube::util::fetch-swagger-spec

kube::log::status "SUCCESS"

# ex: ts=2 sw=2 et filetype=sh
