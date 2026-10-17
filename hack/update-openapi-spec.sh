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
DISCOVERY_ROOT_DIR="${KUBE_ROOT}/api/discovery"
OPENAPI_ROOT_DIR="${KUBE_ROOT}/api/openapi-spec"
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::util::require-jq
kube::golang::setup_env
kube::etcd::install

# We need to call `make` here because that includes all of the compile and link
# flags that we use for a production build, which we need for this script.
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

TMP_DIR=${TMP_DIR:-$(kube::realpath "$(mktemp -d -t "$(basename "$0").XXXXXX")")}
ETCD_HOST=${ETCD_HOST:-127.0.0.1}
ETCD_PORT=${ETCD_PORT:-2379}
API_PORT=${API_PORT:-8050}
API_HOST=${API_HOST:-127.0.0.1}
API_LOGFILE=${API_LOGFILE:-${TMP_DIR}/openapi-api-server.log}

# Omit enums from the spec, which is used to generate clients, until
# https://github.com/kubernetes/kubernetes/issues/109177 is resolved.
# Set KUBE_OPENAPI_SPEC_KEEP_ENUMS=true to keep them.
KUBE_OPENAPI_SPEC_KEEP_ENUMS=${KUBE_OPENAPI_SPEC_KEEP_ENUMS:-false}
OPENAPI_ENUM_FILTER='.'
if [[ "${KUBE_OPENAPI_SPEC_KEEP_ENUMS}" != "true" ]]; then
  # A non-array "enum" is a JSONSchemaProps property, not enum values.
  OPENAPI_ENUM_FILTER='walk(
    if type == "object" then
      (if (.enum | type) == "array" then del(.enum) else . end)
      | if (.description | type) == "string" then
          .description |= sub("\\s*Possible enum values:[\\s\\S]*$"; "")
        else . end
    else . end)'
fi

kube::etcd::start

echo "dummy_token,admin,admin" > "${TMP_DIR}/tokenauth.csv"

# setup envs for TokenRequest required flags
SERVICE_ACCOUNT_LOOKUP=${SERVICE_ACCOUNT_LOOKUP:-true}
SERVICE_ACCOUNT_KEY=${SERVICE_ACCOUNT_KEY:-${TMP_DIR}/kube-serviceaccount.key}
# Generate ServiceAccount key if needed
if [[ ! -f "${SERVICE_ACCOUNT_KEY}" ]]; then
  mkdir -p "$(dirname "${SERVICE_ACCOUNT_KEY}")"
  openssl genrsa -out "${SERVICE_ACCOUNT_KEY}" 2048 2>/dev/null
fi

# Start kube-apiserver
kube::log::status "Starting kube-apiserver"
# KUBE_APISERVER_STRICT_REMOVED_API_HANDLING_IN_ALPHA ensures that the OpenAPI is updated with all APIs
# that are intended to be removed at a particular release during alpha. 
# If a new version tag was just created and you are seeing an unrelated diff when adding
# a new API, run `KUBE_APISERVER_STRICT_REMOVED_API_HANDLING_IN_ALPHA=false ./hack/update-openapi-spec.sh`.
KUBE_APISERVER_STRICT_REMOVED_API_HANDLING_IN_ALPHA=${KUBE_APISERVER_STRICT_REMOVED_API_HANDLING_IN_ALPHA:-true} kube-apiserver \
  --bind-address="${API_HOST}" \
  --secure-port="${API_PORT}" \
  --etcd-servers="http://${ETCD_HOST}:${ETCD_PORT}" \
  --advertise-address="10.10.10.10" \
  --cert-dir="${TMP_DIR}/certs" \
  --feature-gates=AllAlpha=true,AllBeta=true \
  --runtime-config="api/all=true" \
  --token-auth-file="${TMP_DIR}/tokenauth.csv" \
  --authorization-mode=RBAC \
  --service-account-key-file="${SERVICE_ACCOUNT_KEY}" \
  --service-account-lookup="${SERVICE_ACCOUNT_LOOKUP}" \
  --service-account-issuer="https://kubernetes.default.svc" \
  --service-account-signing-key-file="${SERVICE_ACCOUNT_KEY}" \
  --enable-logs-handler=true \
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

kube::log::status "Updating aggregated discovery"

rm -fr "${DISCOVERY_ROOT_DIR}"
mkdir -p "${DISCOVERY_ROOT_DIR}"
curl -kfsS -H 'Authorization: Bearer dummy_token' -H 'Accept: application/json;g=apidiscovery.k8s.io;v=v2;as=APIGroupDiscoveryList' "https://${API_HOST}:${API_PORT}/apis" | jq -S . > "${DISCOVERY_ROOT_DIR}/aggregated_v2.json"

kube::log::status "Updating " "${OPENAPI_ROOT_DIR} for OpenAPI v2"

rm -f "${OPENAPI_ROOT_DIR}/swagger.json"
curl -w "\n" -kfsS -H 'Authorization: Bearer dummy_token' \
  "https://${API_HOST}:${API_PORT}/openapi/v2" \
  | jq -S '.info.version="unversioned" | '"${OPENAPI_ENUM_FILTER}" \
  > "${OPENAPI_ROOT_DIR}/swagger.json"

kube::log::status "Updating " "${OPENAPI_ROOT_DIR}/v3 for OpenAPI v3"

mkdir -p "${OPENAPI_ROOT_DIR}/v3"
# clean up folder, note that some files start with dot like
# ".well-known__openid-configuration_openapi.json"
rm -r "${OPENAPI_ROOT_DIR}"/v3/{*,.*} || true

rm -rf "${OPENAPI_ROOT_DIR}/v3/*"
curl -w "\n" -kfsS -H 'Authorization: Bearer dummy_token' \
  "https://${API_HOST}:${API_PORT}/openapi/v3" \
  | jq -r '.paths | to_entries | .[].key' \
  | while read -r group; do
    kube::log::status "Updating OpenAPI spec and discovery for group ${group}"
    OPENAPI_FILENAME="${group}_openapi.json"
    OPENAPI_FILENAME_ESCAPED="${OPENAPI_FILENAME//\//__}"
    OPENAPI_PATH="${OPENAPI_ROOT_DIR}/v3/${OPENAPI_FILENAME_ESCAPED}"
    curl -w "\n" -kfsS -H 'Authorization: Bearer dummy_token' \
      "https://${API_HOST}:${API_PORT}/openapi/v3/{$group}" \
      | jq -S '.info.version="unversioned" | '"${OPENAPI_ENUM_FILTER}" \
      > "$OPENAPI_PATH"

    if [[ "${group}" == "api"* ]]; then
      DISCOVERY_FILENAME="${group}.json"
      DISCOVERY_FILENAME_ESCAPED="${DISCOVERY_FILENAME//\//__}"
      DISCOVERY_PATH="${DISCOVERY_ROOT_DIR}/${DISCOVERY_FILENAME_ESCAPED}"
      curl -kfsS -H 'Authorization: Bearer dummy_token' "https://${API_HOST}:${API_PORT}/{$group}" | jq -S . > "$DISCOVERY_PATH"
    fi
done

kube::log::status "SUCCESS"

# ex: ts=2 sw=2 et filetype=sh
