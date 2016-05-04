#!/bin/bash

# Copyright 2015 The Kubernetes Authors All rights reserved.
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
# Puts the updated spec at swagger-spec/

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

"${KUBE_ROOT}/hack/build-go.sh" cmd/kube-apiserver

function cleanup()
{
    [[ -n ${APISERVER_PID-} ]] && kill ${APISERVER_PID} 1>&2 2>/dev/null

    kube::etcd::cleanup

    kube::log::status "Clean up complete"
}

trap cleanup EXIT SIGINT

kube::golang::setup_env

apiserver=$(kube::util::find-binary "kube-apiserver")

TMP_DIR=$(mktemp -d /tmp/update-swagger-spec.XXXX)
ETCD_HOST=${ETCD_HOST:-127.0.0.1}
ETCD_PORT=${ETCD_PORT:-4001}
API_PORT=${API_PORT:-8050}
API_HOST=${API_HOST:-127.0.0.1}
KUBELET_PORT=${KUBELET_PORT:-10250}

kube::etcd::start

# Start kube-apiserver
kube::log::status "Starting kube-apiserver"
"${KUBE_OUTPUT_HOSTBIN}/kube-apiserver" \
  --insecure-bind-address="127.0.0.1" \
  --bind-address="127.0.0.1" \
  --insecure-port="${API_PORT}" \
  --etcd-servers="http://${ETCD_HOST}:${ETCD_PORT}" \
  --advertise-address="10.10.10.10" \
  --cert-dir="${TMP_DIR}/certs" \
  --service-cluster-ip-range="10.0.0.0/24" >/tmp/swagger-api-server.log 2>&1 &
APISERVER_PID=$!

kube::util::wait_for_url "http://127.0.0.1:${API_PORT}/healthz" "apiserver: "

SWAGGER_API_PATH="http://127.0.0.1:${API_PORT}/swaggerapi/"
DEFAULT_GROUP_VERSIONS="v1 autoscaling/v1 batch/v1 extensions/v1beta1 apps/v1alpha1"
VERSIONS=${VERSIONS:-$DEFAULT_GROUP_VERSIONS}

kube::log::status "Updating " ${SWAGGER_ROOT_DIR}

for ver in ${VERSIONS}; do
  # fetch the swagger spec for each group version.
  if [[ ${ver} == "v1" ]]; then
    SUBPATH="api"
  else
    SUBPATH="apis"
  fi
  SUBPATH="${SUBPATH}/${ver}"
  SWAGGER_JSON_NAME="$(kube::util::gv-to-swagger-name ${ver}).json"
  curl -w "\n" -fs "${SWAGGER_API_PATH}${SUBPATH}" > "${SWAGGER_ROOT_DIR}/${SWAGGER_JSON_NAME}"

  # fetch the swagger spec for the discovery mechanism at group level.
  if [[ ${ver} == "v1" ]]; then
    continue
  fi
  SUBPATH="apis/"${ver%/*}
  SWAGGER_JSON_NAME="${ver%/*}.json"
  curl -w "\n" -fs "${SWAGGER_API_PATH}${SUBPATH}" > "${SWAGGER_ROOT_DIR}/${SWAGGER_JSON_NAME}"
done

# fetch swagger specs for other discovery mechanism.
curl -w "\n" -fs "${SWAGGER_API_PATH}" > "${SWAGGER_ROOT_DIR}/resourceListing.json"
curl -w "\n" -fs "${SWAGGER_API_PATH}version" > "${SWAGGER_ROOT_DIR}/version.json"
curl -w "\n" -fs "${SWAGGER_API_PATH}api" > "${SWAGGER_ROOT_DIR}/api.json"
curl -w "\n" -fs "${SWAGGER_API_PATH}apis" > "${SWAGGER_ROOT_DIR}/apis.json"
kube::log::status "SUCCESS"

# ex: ts=2 sw=2 et filetype=sh
