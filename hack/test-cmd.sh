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

# This command checks that the built commands can function together for
# simple scenarios.  It does not require Docker so it can run in travis.

source $(dirname $0)/util.sh

function cleanup()
{
    set +e
    kill ${APISERVER_PID} 1>&2 2>/dev/null
    kill ${CTLRMGR_PID} 1>&2 2>/dev/null
    kill ${KUBELET_PID} 1>&2 2>/dev/null
    kill ${PROXY_PID} 1>&2 2>/dev/null
    kill ${ETCD_PID} 1>&2 2>/dev/null
    rm -rf ${ETCD_DIR} 1>&2 2>/dev/null
    echo
    echo "Complete"
}

trap cleanup EXIT SIGINT

set -e

# Start etcd
start_etcd

ETCD_HOST=${ETCD_HOST:-127.0.0.1}
ETCD_PORT=${ETCD_PORT:-4001}
API_PORT=${API_PORT:-8080}
API_HOST=${API_HOST:-127.0.0.1}
KUBELET_PORT=${KUBELET_PORT:-10250}
GO_OUT=$(dirname $0)/../_output/go/bin

# Check kubecfg
out=$(${GO_OUT}/kubecfg -version)
echo kubecfg: $out

# Start kubelet
${GO_OUT}/kubelet \
  --etcd_servers="http://${ETCD_HOST}:${ETCD_PORT}" \
  --hostname_override="127.0.0.1" \
  --address="127.0.0.1" \
  --port="$KUBELET_PORT" 1>&2 &
KUBELET_PID=$!

wait_for_url "http://127.0.0.1:${KUBELET_PORT}/healthz" "kubelet: "

# Start apiserver
${GO_OUT}/apiserver \
  --address="127.0.0.1" \
  --port="${API_PORT}" \
  --etcd_servers="http://${ETCD_HOST}:${ETCD_PORT}" \
  --machines="127.0.0.1" \
  --minion_port=${KUBELET_PORT} 1>&2 &
APISERVER_PID=$!

wait_for_url "http://127.0.0.1:${API_PORT}/healthz" "apiserver: "

KUBE_CMD="${GO_OUT}/kubecfg -h http://127.0.0.1:${API_PORT} -expect_version_match"

${KUBE_CMD} list pods
echo "kubecfg(pods): ok"

${KUBE_CMD} list services
${KUBE_CMD} -c examples/guestbook/frontend-service.json create services
${KUBE_CMD} delete services/frontend
echo "kubecfg(services): ok"

${KUBE_CMD} list minions
${KUBE_CMD} get minions/127.0.0.1
echo "kubecfg(minions): ok"

# Start controller manager
#${GO_OUT}/controller-manager \
#  --etcd_servers="http://127.0.0.1:${ETCD_PORT}" \
#  --master="127.0.0.1:${API_PORT}" 1>&2 &
#CTLRMGR_PID=$!

# Start proxy
#PROXY_LOG=/tmp/kube-proxy.log
#${GO_OUT}/proxy \
#  --etcd_servers="http://127.0.0.1:${ETCD_PORT}" 1>&2 &
#PROXY_PID=$!
