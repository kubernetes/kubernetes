#!/bin/bash

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

# Script to fetch latest swagger spec from federation-apiserver
# Puts the updated spec at federation/apis/swagger-spec/

set -o errexit
set -o nounset
set -o pipefail

cat << __EOF__
Note: This assumes that the 'types_swagger_doc_generated.go' file has been
updated for all API group versions. If you are unsure, please run
hack/update-generated-swagger-docs.sh and
hack/update-federation-generated-swagger-docs.sh first.
__EOF__

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
SWAGGER_ROOT_DIR="${KUBE_ROOT}/federation/apis/swagger-spec"
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::golang::setup_env

make -C "${KUBE_ROOT}" WHAT="cmd/hyperkube"

function cleanup()
{
    [[ -n ${APISERVER_PID-} ]] && kill ${APISERVER_PID} 1>&2 2>/dev/null

    kube::etcd::cleanup

    kube::log::status "Clean up complete"
}

trap cleanup EXIT SIGINT

kube::golang::setup_env

TMP_DIR=$(mktemp -d /tmp/update-federation-swagger-spec.XXXX)
ETCD_HOST=${ETCD_HOST:-127.0.0.1}
ETCD_PORT=${ETCD_PORT:-2379}
API_PORT=${API_PORT:-8050}
API_HOST=${API_HOST:-127.0.0.1}

kube::etcd::start

# Start federation-apiserver
kube::log::status "Starting federation-apiserver"
"${KUBE_OUTPUT_HOSTBIN}/hyperkube" federation-apiserver \
  --insecure-bind-address="${API_HOST}" \
  --bind-address="${API_HOST}" \
  --insecure-port="${API_PORT}" \
  --etcd-servers="http://${ETCD_HOST}:${ETCD_PORT}" \
  --advertise-address="10.10.10.10" \
  --cert-dir="${TMP_DIR}/certs"  >/tmp/swagger-federation-api-server.log 2>&1 &
APISERVER_PID=$!

kube::util::wait_for_url "${API_HOST}:${API_PORT}/" "apiserver: "

SWAGGER_API_PATH="${API_HOST}:${API_PORT}/swaggerapi/"
DEFAULT_GROUP_VERSIONS="v1 extensions/v1beta1 federation/v1beta1"
VERSIONS=${VERSIONS:-$DEFAULT_GROUP_VERSIONS}

kube::log::status "Updating " ${SWAGGER_ROOT_DIR}

SWAGGER_API_PATH="${SWAGGER_API_PATH}" SWAGGER_ROOT_DIR="${SWAGGER_ROOT_DIR}" VERSIONS="${VERSIONS}" kube::util::fetch-swagger-spec

kube::log::status "SUCCESS"

# ex: ts=2 sw=2 et filetype=sh
