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

if [ "$(which etcd)" == "" ]; then
	echo "etcd must be in your PATH"
	exit 1
fi

# Stop right away if the build fails
set -e

$(dirname $0)/build-go.sh

echo "Starting etcd"

ETCD_DIR=$(mktemp -d -t kube-integration.XXXXXX)

(etcd -name test -data-dir ${ETCD_DIR} &> /tmp/etcd.log) &
ETCD_PID=$!

sleep 5

# Shut down anyway if there's an error.
set +e

API_PORT=8080
KUBELET_PORT=10250

GO_OUT=$(dirname $0)/../output/go

APISERVER_LOG=/tmp/apiserver.log
${GO_OUT}/apiserver \
  --address="127.0.0.1" \
  --port="${API_PORT}" \
  --etcd_servers="http://127.0.0.1:4001" \
  --machines="127.0.0.1" &> ${APISERVER_LOG} &
APISERVER_PID=$!

CTLRMGR_LOG=/tmp/controller-manager.log
${GO_OUT}/controller-manager \
  --etcd_servers="http://127.0.0.1:4001" \
  --master="127.0.0.1:${API_PORT}" &> ${CTLRMGR_LOG} &
CTLRMGR_PID=$!

KUBELET_LOG=/tmp/kubelet.log
${GO_OUT}/kubelet \
  --etcd_servers="http://127.0.0.1:4001" \
  --hostname_override="127.0.0.1" \
  --address="127.0.0.1" \
  --port="$KUBELET_PORT" &> ${KUBELET_LOG} &
KUBELET_PID=$!


echo "Local Kubernetes cluster is running. Press Ctrl-C to shut it down."
echo "Logs: "
echo "  ${APISERVER_LOG}"
echo "  ${CTLRMGR_LOG}"
echo "  ${KUBELET_LOG}"

cleanup()
{
    echo "Cleaning up..."
    kill ${APISERVER_PID}
    kill ${CTLRMGR_PID}
    kill ${KUBELET_PID}

    kill ${ETCD_PID}
    rm -rf ${ETCD_DIR}
    exit 0
}
 
trap cleanup EXIT

while true; do read x; done
