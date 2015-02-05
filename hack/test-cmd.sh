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

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

function cleanup()
{
    [[ -n ${APISERVER_PID-} ]] && kill ${APISERVER_PID} 1>&2 2>/dev/null
    [[ -n ${CTLRMGR_PID-} ]] && kill ${CTLRMGR_PID} 1>&2 2>/dev/null
    [[ -n ${KUBELET_PID-} ]] && kill ${KUBELET_PID} 1>&2 2>/dev/null
    [[ -n ${PROXY_PID-} ]] && kill ${PROXY_PID} 1>&2 2>/dev/null

    kube::etcd::cleanup

    kube::log::status "Clean up complete"
}

trap cleanup EXIT SIGINT

kube::etcd::start

ETCD_HOST=${ETCD_HOST:-127.0.0.1}
ETCD_PORT=${ETCD_PORT:-4001}
API_PORT=${API_PORT:-8080}
API_HOST=${API_HOST:-127.0.0.1}
KUBELET_PORT=${KUBELET_PORT:-10250}
CTLRMGR_PORT=${CTLRMGR_PORT:-10252}

# Check kubectl
kube::log::status "Running kubectl with no options"
"${KUBE_OUTPUT_HOSTBIN}/kubectl"

kube::log::status "Starting kubelet in masterless mode"
"${KUBE_OUTPUT_HOSTBIN}/kubelet" \
  --really_crash_for_testing=true \
  --root_dir=/tmp/kubelet.$$ \
  --docker_endpoint="fake://" \
  --address="127.0.0.1" \
  --port="$KUBELET_PORT" 1>&2 &
KUBELET_PID=$!
kube::util::wait_for_url "http://127.0.0.1:${KUBELET_PORT}/healthz" "kubelet: "
kill ${KUBELET_PID} 1>&2 2>/dev/null

kube::log::status "Starting kubelet in masterful mode"
"${KUBE_OUTPUT_HOSTBIN}/kubelet" \
  --really_crash_for_testing=true \
  --root_dir=/tmp/kubelet.$$ \
  --docker_endpoint="fake://" \
  --etcd_servers="http://${ETCD_HOST}:${ETCD_PORT}" \
  --hostname_override="127.0.0.1" \
  --address="127.0.0.1" \
  --api_servers="${API_HOST}:${API_PORT}" \
  --auth_path="${KUBE_ROOT}/hack/.test-cmd-auth" \
  --port="$KUBELET_PORT" 1>&2 &
KUBELET_PID=$!

kube::util::wait_for_url "http://127.0.0.1:${KUBELET_PORT}/healthz" "kubelet: "

# Start kube-apiserver
kube::log::status "Starting kube-apiserver"
"${KUBE_OUTPUT_HOSTBIN}/kube-apiserver" \
  --address="127.0.0.1" \
  --public_address_override="127.0.0.1" \
  --port="${API_PORT}" \
  --etcd_servers="http://${ETCD_HOST}:${ETCD_PORT}" \
  --public_address_override="127.0.0.1" \
  --kubelet_port=${KUBELET_PORT} \
  --runtime_config=api/v1beta3 \
  --portal_net="10.0.0.0/24" 1>&2 &
APISERVER_PID=$!

kube::util::wait_for_url "http://127.0.0.1:${API_PORT}/healthz" "apiserver: "

# Start controller manager
kube::log::status "Starting CONTROLLER-MANAGER"
"${KUBE_OUTPUT_HOSTBIN}/kube-controller-manager" \
  --machines="127.0.0.1" \
  --master="127.0.0.1:${API_PORT}" 1>&2 &
CTLRMGR_PID=$!

kube::util::wait_for_url "http://127.0.0.1:${CTLRMGR_PORT}/healthz" "controller-manager: "
kube::util::wait_for_url "http://127.0.0.1:${API_PORT}/api/v1beta1/minions/127.0.0.1" "apiserver(minions): " 0.2 25

# expose kubectl directly for readability
PATH="${KUBE_OUTPUT_HOSTBIN}":$PATH

kube_api_versions=(
  ""
  v1beta1
  v1beta2
  v1beta3
)
for version in "${kube_api_versions[@]}"; do
  if [[ -z "${version}" ]]; then
    kube_flags=(
      -s "http://127.0.0.1:${API_PORT}"
      --match-server-version
    )
    [ "$(kubectl get minions -t $'{{ .apiVersion }}' "${kube_flags[@]}")" == "v1beta1" ]
  else
    kube_flags=(
      -s "http://127.0.0.1:${API_PORT}"
      --match-server-version
      --api-version="${version}"
    )
    [ "$(kubectl get minions -t $'{{ .apiVersion }}' "${kube_flags[@]}")" == "${version}" ]
  fi

  # passing no arguments to create is an error
  [ ! $(kubectl create) ]

  kube::log::status "Testing kubectl(${version}:pods)"
  kubectl get pods "${kube_flags[@]}"
  kubectl create -f examples/guestbook/redis-master.json "${kube_flags[@]}"
  kubectl get pods "${kube_flags[@]}"
  kubectl get pod redis-master "${kube_flags[@]}"
  [ "$(kubectl get pod redis-master -o template --output-version=v1beta1 -t '{{ .id }}' "${kube_flags[@]}")" == "redis-master" ]
  output_pod=$(kubectl get pod redis-master -o yaml --output-version=v1beta1 "${kube_flags[@]}")
  kubectl delete pod redis-master "${kube_flags[@]}"
  before="$(kubectl get pods -o template -t "{{ len .items }}" "${kube_flags[@]}")"
  echo "${output_pod}" | kubectl create -f - "${kube_flags[@]}"
  after="$(kubectl get pods -o template -t "{{ len .items }}" "${kube_flags[@]}")"
  [ "$((${after} - ${before}))" -eq 1 ]
  kubectl get pods -o yaml --output-version=v1beta1 "${kube_flags[@]}" | grep -q "id: redis-master"
  kubectl describe pod redis-master "${kube_flags[@]}" | grep -q 'Name:.*redis-master'
  kubectl delete -f examples/guestbook/redis-master.json "${kube_flags[@]}"

  kube::log::status "Testing kubectl(${version}:services)"
  kubectl get services "${kube_flags[@]}"
  kubectl create -f examples/guestbook/frontend-service.json "${kube_flags[@]}"
  kubectl get services "${kube_flags[@]}"
  output_service=$(kubectl get service frontend -o json --output-version=v1beta3 "${kube_flags[@]}")
  kubectl delete service frontend "${kube_flags[@]}"
  echo "${output_service}" | kubectl create -f - "${kube_flags[@]}"
  kubectl create -f - "${kube_flags[@]}" << __EOF__
      {
          "kind": "Service",
          "apiVersion": "v1beta1",
          "id": "service-${version}-test",
          "port": 80,
          "protocol": "TCP"
      }
__EOF__
  kubectl get services "${kube_flags[@]}"
  kubectl get services "service-${version}-test" "${kube_flags[@]}"
  kubectl delete service frontend "${kube_flags[@]}"

  kube::log::status "Testing kubectl(${version}:replicationcontrollers)"
  kubectl get replicationcontrollers "${kube_flags[@]}"
  kubectl create -f examples/guestbook/frontend-controller.json "${kube_flags[@]}"
  kubectl get replicationcontrollers "${kube_flags[@]}"
  kubectl describe replicationcontroller frontend-controller "${kube_flags[@]}" | grep -q 'Replicas:.*3 desired'
  kubectl delete rc frontend-controller "${kube_flags[@]}"

  kube::log::status "Testing kubectl(${version}:nodes)"
  kubectl get nodes "${kube_flags[@]}"
  kubectl describe nodes 127.0.0.1 "${kube_flags[@]}"

  if [[ "${version}" != "v1beta3" ]]; then
    kube::log::status "Testing kubectl(${version}:minions)"
    kubectl get minions "${kube_flags[@]}"
    kubectl get minions 127.0.0.1 "${kube_flags[@]}"
    kubectl get minions -o template -t $'{{range.items}}{{.id}}\n{{end}}' "${kube_flags[@]}"
    # TODO: I should be a MinionList instead of List
    [ "$(kubectl get minions -t $'{{ .kind }}' "${kube_flags[@]}")" == "List" ]
  fi
done

kube::log::status "TEST PASSED"
