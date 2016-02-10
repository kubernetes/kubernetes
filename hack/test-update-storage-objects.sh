#!/bin/bash

# Copyright 2014 The Kubernetes Authors All rights reserved.
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

# The api version in which objects are currently stored in etcd.
KUBE_OLD_API_VERSION=${KUBE_OLD_API_VERSION:-"v1"}
# The api version in which our etcd objects should be converted to.
# The new api version
KUBE_NEW_API_VERSION=${KUBE_NEW_API_VERSION:-"v1"}

KUBE_OLD_STORAGE_VERSIONs=${KUBE_OLD_STORAGE_VERSIONs:-""}
KUBE_NEW_STORAGE_VERSIONs=${KUBE_NEW_STORAGE_VERSIONs:-""}

ETCD_HOST=${ETCD_HOST:-127.0.0.1}
ETCD_PORT=${ETCD_PORT:-4001}
API_PORT=${API_PORT:-8080}
API_HOST=${API_HOST:-127.0.0.1}
KUBE_API_VERSIONS=""
RUNTIME_CONFIG=""

KUBECTL="${KUBE_OUTPUT_HOSTBIN}/kubectl"
UPDATE_ETCD_OBJECTS_SCRIPT="${KUBE_ROOT}/cluster/update-storage-objects.sh"

function startApiServer() {
  local storage_versions=${1:-""}
  kube::log::status "Starting kube-apiserver with KUBE_API_VERSIONS: ${KUBE_API_VERSIONS} and runtime-config: ${RUNTIME_CONFIG} and "
  kube::log::status "                            and runtime-config: ${RUNTIME_CONFIG}"
  kube::log::status "                 and storage-version overrides: ${storage_versions}"

  KUBE_API_VERSIONS="${KUBE_API_VERSIONS}" \
    "${KUBE_OUTPUT_HOSTBIN}/kube-apiserver" \
    --insecure-bind-address="127.0.0.1" \
    --bind-address="127.0.0.1" \
    --insecure-port="${API_PORT}" \
    --etcd-servers="http://${ETCD_HOST}:${ETCD_PORT}" \
    --runtime-config="${RUNTIME_CONFIG}" \
    --cert-dir="${TMPDIR:-/tmp/}" \
    --service-cluster-ip-range="10.0.0.0/24" \
    --storage-versions="${storage_versions}" 1>&2 &
  APISERVER_PID=$!

  # url, prefix, wait, times
  kube::util::wait_for_url "http://127.0.0.1:${API_PORT}/healthz" "apiserver: " 1 120
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

kube::etcd::start

function run-upgrade-test() {
  local resource=$1
  local source_file=$2
  kube::log::status "Running test for update etcd object scenario: ${resource}"

  "${KUBE_ROOT}/hack/build-go.sh" cmd/kube-apiserver


  #######################################################
  # Step 1: Start a server which supports both the old and new api versions,
  # but KUBE_OLD_API_VERSION is the latest (storage) version.
  #######################################################

  KUBE_API_VERSIONS="${KUBE_OLD_API_VERSION},${KUBE_NEW_API_VERSION}"
  RUNTIME_CONFIG="api/all=false,api/${KUBE_OLD_API_VERSION}=true,api/${KUBE_NEW_API_VERSION}=true"
  startApiServer ${KUBE_OLD_STORAGE_VERSIONS}

  # Create an object
  kube::log::status "Creating a ${resource}"
  ${KUBECTL} create -f "${source_file}"

  killApiServer


  #######################################################
  # Step 2: Start a server which supports both the old and new api versions,
  # but KUBE_NEW_API_VERSION is the latest (storage) version.
  #######################################################

  KUBE_API_VERSIONS="${KUBE_NEW_API_VERSION},${KUBE_OLD_API_VERSION}"
  RUNTIME_CONFIG="api/all=false,api/${KUBE_OLD_API_VERSION}=true,api/${KUBE_NEW_API_VERSION}=true"
  startApiServer ${KUBE_OLD_STORAGE_VERSIONS}

  # Update etcd objects, so that will now be stored in the new api version.
  ${UPDATE_ETCD_OBJECTS_SCRIPT}

  killApiServer



  #######################################################
  # Step 3 : Start a server which supports only the new api version.
  #######################################################

  KUBE_API_VERSIONS="${KUBE_NEW_API_VERSION}"
  RUNTIME_CONFIG="api/all=false,api/${KUBE_NEW_API_VERSION}=true"

  # This seems to reduce flakiness.
  sleep 1
  startApiServer ${KUBE_NEW_STORAGE_VERSIONS}

  # Verify that the server is able to read the object.
  # This will fail if the object is in a version that is not understood by the
  # master.
  ${KUBECTL} get "${resource}"
}

# run-upgrade-test "pods" "docs/user-guide/pod.yaml"


# NOTE: The KUBE_NEW_API_VERSION includes extensions because batch doesn't work
# if extensions isn't registered, since it's using extension's internal version.
KUBE_OLD_API_VERSION="v1,extensions/v1beta1" \
  KUBE_NEW_API_VERSION="v1,extensions/v1beta1,batch/v1" \
  KUBE_OLD_STORAGE_VERSIONS="batch=extensions/v1beta1" \
  KUBE_NEW_STORAGE_VERSIONS="batch/v1" \
  run-upgrade-test "jobs" "docs/user-guide/job.yaml"
