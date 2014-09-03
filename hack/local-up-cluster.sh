#!/bin/bash

# Copyright 2014 Google Inc. All rights reserved.
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

# This command builds and runs a local kubernetes cluster. It's just like
# local-up.sh, but this one launches the three separate binaries.
# You may need to run this as root to allow kubelet to open docker's socket.

source $(dirname $0)/util.sh

docker ps 2> /dev/null 1> /dev/null
if [ "$?" != "0" ]; then
  echo "Failed to successfully run 'docker ps', please verify that docker is installed and \$DOCKER_HOST is set correctly."
  exit 1
fi

# Stop right away if the build fails
set -e

$(dirname $0)/build-go.sh

echo "Starting etcd"
start_etcd

# Shut down anyway if there's an error.
set +e

API_PORT=${API_PORT:-8080}
API_HOST=${API_HOST:-127.0.0.1}
# By default only allow CORS for requests on localhost
API_CORS_ALLOWED_ORIGINS=${API_CORS_ALLOWED_ORIGINS:-127.0.0.1:.*,localhost:.*}
KUBELET_PORT=${KUBELET_PORT:-10250}

GO_OUT=$(dirname $0)/../_output/go/bin

APISERVER_LOG=/tmp/apiserver.log
"${GO_OUT}/apiserver" \
  --address="${API_HOST}" \
  --port="${API_PORT}" \
  --etcd_servers="http://127.0.0.1:4001" \
  --machines="127.0.0.1" \
  --cors_allowed_origins="${API_CORS_ALLOWED_ORIGINS}" >"${APISERVER_LOG}" 2>&1 &
APISERVER_PID=$!

CTLRMGR_LOG=/tmp/controller-manager.log
"${GO_OUT}/controller-manager" \
  --master="${API_HOST}:${API_PORT}" >"${CTLRMGR_LOG}" 2>&1 &
CTLRMGR_PID=$!

KUBELET_LOG=/tmp/kubelet.log
"${GO_OUT}/kubelet" \
  --etcd_servers="http://127.0.0.1:4001" \
  --hostname_override="127.0.0.1" \
  --address="127.0.0.1" \
  --port="$KUBELET_PORT" >"${KUBELET_LOG}" 2>&1 &
KUBELET_PID=$!

PROXY_LOG=/tmp/kube-proxy.log
"${GO_OUT}/proxy" \
  --master="http://${API_HOST}:${API_PORT}" >"${PROXY_LOG}" 2>&1 &
PROXY_PID=$!

SCHEDULER_LOG=/tmp/k8s-scheduler.log
"${GO_OUT}/scheduler" \
  --master="http://${API_HOST}:${API_PORT}" >"${SCHEDULER_LOG}" 2>&1 &
SCHEDULER_PID=$!

echo "Local Kubernetes cluster is running. Press Ctrl-C to shut it down."
echo "Logs: "
echo "  ${APISERVER_LOG}"
echo "  ${CTLRMGR_LOG}"
echo "  ${KUBELET_LOG}"
echo "  ${PROXY_LOG}"
echo "  ${SCHEDULER_LOG}"

cleanup()
{
    echo "Cleaning up..."
    kill "${APISERVER_PID}"
    kill "${CTLRMGR_PID}"
    kill "${KUBELET_PID}"
    kill "${PROXY_PID}"
    kill "${SCHEDULER_PID}"

    kill "${ETCD_PID}"
    rm -rf "${ETCD_DIR}"
    exit 0
}

trap cleanup EXIT

while true; do read x; done
