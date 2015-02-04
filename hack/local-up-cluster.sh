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
DOCKER_OPTS=${DOCKER_OPTS:-""}
DOCKER_NATIVE=${DOCKER_NATIVE:-""}
DOCKER=(docker ${DOCKER_OPTS})

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
cd "${KUBE_ROOT}"

# Stop right away if the build fails
set -e

source "${KUBE_ROOT}/hack/lib/init.sh"
"${KUBE_ROOT}/hack/build-go.sh"

${DOCKER[@]} ps 2> /dev/null 1> /dev/null
if [ "$?" != "0" ]; then
  echo "Failed to successfully run 'docker ps', please verify that docker is installed and \$DOCKER_HOST is set correctly."
  exit 1
fi

echo "Starting etcd"
kube::etcd::start

# Shut down anyway if there's an error.
set +e

API_PORT=${API_PORT:-8080}
API_HOST=${API_HOST:-127.0.0.1}
# By default only allow CORS for requests on localhost
API_CORS_ALLOWED_ORIGINS=${API_CORS_ALLOWED_ORIGINS:-"/127.0.0.1(:[0-9]+)?$,/localhost(:[0-9]+)?$"}
KUBELET_PORT=${KUBELET_PORT:-10250}
LOG_LEVEL=${LOG_LEVEL:-3}

# Detect the OS name/arch so that we can find our binary
case "$(uname -s)" in
  Darwin)
    host_os=darwin
    ;;
  Linux)
    host_os=linux
    ;;
  *)
    echo "Unsupported host OS.  Must be Linux or Mac OS X." >&2
    exit 1
    ;;
esac

case "$(uname -m)" in
  x86_64*)
    host_arch=amd64
    ;;
  i?86_64*)
    host_arch=amd64
    ;;
  amd64*)
    host_arch=amd64
    ;;
  arm*)
    host_arch=arm
    ;;
  i?86*)
    host_arch=x86
    ;;
  *)
    echo "Unsupported host arch. Must be x86_64, 386 or arm." >&2
    exit 1
    ;;
esac

GO_OUT="${KUBE_ROOT}/_output/local/bin/${host_os}/${host_arch}"

APISERVER_LOG=/tmp/kube-apiserver.log
sudo "${GO_OUT}/kube-apiserver" \
  --v=${LOG_LEVEL} \
  --address="${API_HOST}" \
  --port="${API_PORT}" \
  --runtime_config=api/v1beta3 \
  --etcd_servers="http://localhost:4001" \
  --portal_net="10.0.0.0/24" \
  --cors_allowed_origins="${API_CORS_ALLOWED_ORIGINS}" >"${APISERVER_LOG}" 2>&1 &
APISERVER_PID=$!

# Wait for kube-apiserver to come up before launching the rest of the components.
kube::util::wait_for_url "http://${API_HOST}:${API_PORT}/api/v1beta1/pods" "apiserver: "

CTLRMGR_LOG=/tmp/kube-controller-manager.log
sudo "${GO_OUT}/kube-controller-manager" \
  --v=${LOG_LEVEL} \
  --machines="127.0.0.1" \
  --master="${API_HOST}:${API_PORT}" >"${CTLRMGR_LOG}" 2>&1 &
CTLRMGR_PID=$!

KUBELET_LOG=/tmp/kubelet.log
sudo "${GO_OUT}/kubelet" \
  --v=${LOG_LEVEL} \
  --etcd_servers="http://127.0.0.1:4001" \
  --hostname_override="127.0.0.1" \
  --address="127.0.0.1" \
  --api_servers="${API_HOST}:${API_PORT}" \
  --auth_path="${KUBE_ROOT}/hack/.test-cmd-auth" \
  --port="$KUBELET_PORT" >"${KUBELET_LOG}" 2>&1 &
KUBELET_PID=$!

PROXY_LOG=/tmp/kube-proxy.log
sudo "${GO_OUT}/kube-proxy" \
  --v=${LOG_LEVEL} \
  --master="http://${API_HOST}:${API_PORT}" >"${PROXY_LOG}" 2>&1 &
PROXY_PID=$!

SCHEDULER_LOG=/tmp/kube-scheduler.log
sudo "${GO_OUT}/kube-scheduler" \
  --v=${LOG_LEVEL} \
  --master="http://${API_HOST}:${API_PORT}" >"${SCHEDULER_LOG}" 2>&1 &
SCHEDULER_PID=$!

cat <<EOF
Local Kubernetes cluster is running. Press Ctrl-C to shut it down.

Logs:
  ${APISERVER_LOG}
  ${CTLRMGR_LOG}
  ${KUBELET_LOG}
  ${PROXY_LOG}
  ${SCHEDULER_LOG}

To start using your cluster, open up another terminal/tab and run:

  export KUBERNETES_PROVIDER=local
  cluster/kubectl.sh
EOF

cleanup()
{
    echo "Cleaning up..."
    sudo kill "${APISERVER_PID}"
    sudo kill "${CTLRMGR_PID}"
    sudo kill "${KUBELET_PID}"
    sudo kill "${PROXY_PID}"
    sudo kill "${SCHEDULER_PID}"

    kill "${ETCD_PID}"
    rm -rf "${ETCD_DIR}"
    exit 0
}

trap cleanup EXIT

while true; do sleep 1; done
