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

# Shut down anyway if there's an error.
set +e

API_PORT=${API_PORT:-8080}
API_HOST=${API_HOST:-127.0.0.1}
# By default only allow CORS for requests on localhost
API_CORS_ALLOWED_ORIGINS=${API_CORS_ALLOWED_ORIGINS:-"/127.0.0.1(:[0-9]+)?$,/localhost(:[0-9]+)?$"}
KUBELET_PORT=${KUBELET_PORT:-10250}
LOG_LEVEL=${LOG_LEVEL:-3}
CHAOS_CHANCE=${CHAOS_CHANCE:-0.0}

# For the common local scenario, fail fast if server is already running.
# this can happen if you run local-up-cluster.sh twice and kill etcd in between.
curl $API_HOST:$API_PORT
if [ ! $? -eq 0 ]; then
    echo "API SERVER port is free, proceeding..."
else
    echo "ERROR starting API SERVER, exiting.  Some host on $API_HOST is serving already on $API_PORT"
    exit 1
fi

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

cleanup()
{
    echo "Cleaning up..."
    [[ -n "${APISERVER_PID-}" ]] && sudo kill "${APISERVER_PID}"
    [[ -n "${CTLRMGR_PID-}" ]] && sudo kill "${CTLRMGR_PID}"
    [[ -n "${KUBELET_PID-}" ]] && sudo kill "${KUBELET_PID}"
    [[ -n "${PROXY_PID-}" ]] && sudo kill "${PROXY_PID}"
    [[ -n "${SCHEDULER_PID-}" ]] && sudo kill "${SCHEDULER_PID}"

    [[ -n "${ETCD_PID-}" ]] && kube::etcd::stop
    [[ -n "${ETCD_DIR-}" ]] && kube::etcd::clean_etcd_dir

    exit 0
}

trap cleanup EXIT

echo "Starting etcd"
kube::etcd::start

# Admission Controllers to invoke prior to persisting objects in cluster
ADMISSION_CONTROL=NamespaceLifecycle,NamespaceAutoProvision,LimitRanger,ResourceQuota

APISERVER_LOG=/tmp/kube-apiserver.log
sudo -E "${GO_OUT}/kube-apiserver" \
  --v=${LOG_LEVEL} \
  --admission_control="${ADMISSION_CONTROL}" \
  --address="${API_HOST}" \
  --port="${API_PORT}" \
  --runtime_config=api/v1beta3 \
  --etcd_servers="http://127.0.0.1:4001" \
  --portal_net="10.0.0.0/24" \
  --cors_allowed_origins="${API_CORS_ALLOWED_ORIGINS}" >"${APISERVER_LOG}" 2>&1 &
APISERVER_PID=$!

# Wait for kube-apiserver to come up before launching the rest of the components.
echo "Waiting for apiserver to come up"
kube::util::wait_for_url "http://${API_HOST}:${API_PORT}/api/v1beta3/pods" "apiserver: " 1 10 || exit 1

CTLRMGR_LOG=/tmp/kube-controller-manager.log
sudo -E "${GO_OUT}/kube-controller-manager" \
  --v=${LOG_LEVEL} \
  --machines="127.0.0.1" \
  --master="${API_HOST}:${API_PORT}" >"${CTLRMGR_LOG}" 2>&1 &
CTLRMGR_PID=$!

KUBELET_LOG=/tmp/kubelet.log
sudo -E "${GO_OUT}/kubelet" \
  --v=${LOG_LEVEL} \
  --chaos_chance="${CHAOS_CHANCE}" \
  --hostname_override="127.0.0.1" \
  --address="127.0.0.1" \
  --api_servers="${API_HOST}:${API_PORT}" \
  --auth_path="${KUBE_ROOT}/hack/.test-cmd-auth" \
  --port="$KUBELET_PORT" >"${KUBELET_LOG}" 2>&1 &
KUBELET_PID=$!

PROXY_LOG=/tmp/kube-proxy.log
sudo -E "${GO_OUT}/kube-proxy" \
  --v=${LOG_LEVEL} \
  --master="http://${API_HOST}:${API_PORT}" >"${PROXY_LOG}" 2>&1 &
PROXY_PID=$!

SCHEDULER_LOG=/tmp/kube-scheduler.log
sudo -E "${GO_OUT}/kube-scheduler" \
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

  cluster/kubectl.sh config set-cluster local --server=http://${API_HOST}:${API_PORT} --insecure-skip-tls-verify=true
  cluster/kubectl.sh config set-context local --cluster=local
  cluster/kubectl.sh config use-context local
  cluster/kubectl.sh
EOF

while true; do sleep 1; done
