#!/bin/bash

# Copyright 2018 The Kubernetes Authors.
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

# This script fetches list of latest api resources served by kube apiserver and
# saves the list at test/e2e/testing-manifests/apiresource/resources_all.csv.
# This script also computes the list of api resources used for testing
#   (test/e2e/testing-manifests/apiresource/resources.csv)
# by removing whitelisted api resources
#   (test/e2e/testing-manifests/apiresource/resources_whitelist.csv) from the
# list of all resources
#   (test/e2e/testing-manifests/apiresource/resources_all.csv)
#
# The result list of api resources for testing
#   (test/e2e/testing-manifests/apiresource/resources.csv)
# is used by the api coverage e2e test:
#   test/e2e/apimachinery/coverage.go
# Ref test/e2e/testing-manifests/apiresource/README.md for more information.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::golang::setup_env

make -C "${KUBE_ROOT}" WHAT=cmd/kube-apiserver

function cleanup()
{
    [[ -n ${APISERVER_PID-} ]] && kill ${APISERVER_PID} 1>&2 2>/dev/null

    kube::etcd::cleanup

    kube::log::status "Clean up complete"
}

TMP_DIR=$(mktemp -d /tmp/update-api-resource-list.XXXX)
KUBE_APISERVER_CERT="$TMP_DIR/certs/apiserver.crt"
AUTH_TOKEN_HEADER="Authorization: Bearer dummy_token"

# fetch_api_resource walks through apiserver's API discovery paths
# $APISERVER_URL/api and $APISERVER_URL/apis for each group, version and
# resource. It returns the list of all exposed API resources verbs in csv format
# of:
#   GROUP,VERSION,RESOURCE,NAMESPACED,VERB
#
# GROUP is empty string "" for core group. RESOURCE may contain "/" to indicate
# a subresource. NAMESPACED is either "true" or "false".
# The output is not sorted yet. We have following steps to sort the outputs.
function fetch_api_resource()
{
for RESOURCE in $(curl -s --cacert $KUBE_APISERVER_CERT -H "$AUTH_TOKEN_HEADER" $APISERVER_URL/api/v1 | jq -c .resources[]); do
  NAME=$(echo $RESOURCE | jq .name | tr -d '"')
  NAMESPACED=$(echo $RESOURCE | jq .namespaced)
  for VERB in $(echo $RESOURCE | jq '.verbs' | tr -d '"[],'); do
    echo ,v1,$NAME,$NAMESPACED,$VERB
  done
done

# List group versions
for GROUP_VERSION in $(curl -s --cacert $KUBE_APISERVER_CERT -H "$AUTH_TOKEN_HEADER" $APISERVER_URL/apis | jq .groups[].versions[].groupVersion | tr -d '"'); do
  # List resources
  for RESOURCE in $(curl -s --cacert $KUBE_APISERVER_CERT -H "$AUTH_TOKEN_HEADER" $APISERVER_URL/apis/$GROUP_VERSION | jq -c '.resources[]'); do
    NAME=$(echo $RESOURCE | jq '.name' | tr -d '"')
    NAMESPACED=$(echo $RESOURCE | jq '.namespaced')
    for VERB in $(echo $RESOURCE | jq '.verbs' | tr -d '"[],'); do
      echo $(echo $GROUP_VERSION | tr '/' ','),$NAME,$NAMESPACED,$VERB
    done
  done
done
}

kube::util::trap_add cleanup EXIT SIGINT

kube::golang::setup_env

apiserver=$(kube::util::find-binary "kube-apiserver")

ETCD_HOST=${ETCD_HOST:-127.0.0.1}
ETCD_PORT=${ETCD_PORT:-2379}
API_PORT=${API_PORT:-8050}
API_HOST=${API_HOST:-127.0.0.1}

kube::etcd::start

echo "dummy_token,admin,admin" > $TMP_DIR/tokenauth.csv

# Start kube-apiserver with all apis enabled
kube::log::status "Starting kube-apiserver"
"${KUBE_OUTPUT_HOSTBIN}/kube-apiserver" \
  --bind-address="${API_HOST}" \
  --secure-port="${API_PORT}" \
  --etcd-servers="http://${ETCD_HOST}:${ETCD_PORT}" \
  --advertise-address="10.10.10.10" \
  --cert-dir="${TMP_DIR}/certs" \
  --runtime-config="api/all=true" \
  --token-auth-file=$TMP_DIR/tokenauth.csv \
  --logtostderr \
  --v=2 \
  --service-cluster-ip-range="10.0.0.0/24" >/tmp/openapi-api-server.log 2>&1 &
APISERVER_PID=$!

# wait for apiserver to come up
kube::util::wait_for_url_ssl "https://${API_HOST}:${API_PORT}/healthz" "apiserver: " 1 30 1 --cacert $KUBE_APISERVER_CERT -H "$AUTH_TOKEN_HEADER"

APISERVER_URL="https://${API_HOST}:${API_PORT}"

KUBE_RESOURCE_FILE="${KUBE_ROOT}/test/e2e/testing-manifests/apiresource/resources_all.csv"
KUBE_RESOURCE_WHITELIST_FILE="${KUBE_ROOT}/test/e2e/testing-manifests/apiresource/resources_whitelist.csv"
KUBE_RESOURCE_TEST_FILE="${KUBE_ROOT}/test/e2e/testing-manifests/apiresource/resources.csv"

# For the propose of easily comparing KUBE_RESOURCE_FILE and
# KUBE_RESOURCE_WHITELIST_FILE, in order to remove whitelisted API resource
# lines, we need to sort the files.
#
# The API coverage e2e test requires reading parent resource before reading
# subresource, to correctly construct the resource map. For example we want
#   ,v1,pods,true,VERB
# to be ordered before
#   ,v1,pods/status,true,VERB
fetch_api_resource | sort -t',' -k1,1 -k2,2 -k3,3 -k4 -o ${KUBE_RESOURCE_FILE}
sort -t',' -k1,1 -k2,2 -k3,3 -k4 ${KUBE_RESOURCE_WHITELIST_FILE} -o ${KUBE_RESOURCE_WHITELIST_FILE}
# Remove whitelisted API resource lines from KUBE_RESOURCE_FILE, to generated
# KUBE_RESOURCE_TEST_FILE.
#
# NOTE: GNU comm and BSD comm have different defaulting. We sort the inputs to satisfy
# the defaulting first, then sort the output to more readable csv format
comm -13 <(sort ${KUBE_RESOURCE_WHITELIST_FILE}) <(sort ${KUBE_RESOURCE_FILE}) | sort -t',' -k1,1 -k2,2 -k3,3 -k4 -o ${KUBE_RESOURCE_TEST_FILE}

kube::log::status "WARNING: (please ignore if the file <test/e2e/testing-manifests/apiresource/resources.csv> is not changed)"
kube::log::status "  If the API resource list (test/e2e/testing-manifests/apiresource/resources.csv) gets changed AND/OR if you are making API change that adds/updates some APIs in <GROUP>/<VERSION>/<KIND>, please update the corresponding yamlfiles in test/e2e/testing-manifests/apiresource/yamlfiles/<GROUP>/<VERSION>/<KIND>.yaml to properly pass the API coverage e2e test (test/e2e/apimachinery/coverage.go). For more information, please refer to test/e2e/testing-manifests/apiresource/README.md"
kube::log::status "SUCCESS: API resource list updated"

# ex: ts=2 sw=2 et filetype=sh
