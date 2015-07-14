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

# A set of helpers for starting/running an apiserver for tests

kube::apiserver::start() {
  kube::log::status "Starting kube-apiserver with KUBE_API_VERSIONS: ${KUBE_API_VERSIONS} and runtime_config: ${RUNTIME_CONFIG}"

  KUBE_API_VERSIONS="${KUBE_API_VERSIONS}" \
    "${KUBE_OUTPUT_HOSTBIN}/kube-apiserver" \
    --address="127.0.0.1" \
    --public_address_override="127.0.0.1" \
    --port="${API_PORT}" \
    --etcd_servers="http://${ETCD_HOST}:${ETCD_PORT}" \
    --public_address_override="127.0.0.1" \
    --kubelet_port=${KUBELET_PORT} \
    --runtime_config="${RUNTIME_CONFIG}" \
    --cert_dir="${TMPDIR:-/tmp/}" \
    --portal_net="10.0.0.0/24" 1>&2 &
  APISERVER_PID=$!

  kube::util::wait_for_url "http://127.0.0.1:${API_PORT}/healthz" "apiserver: "
}

kube::apiserver::kill() {
  kube::log::status "Killing api server"
  [[ -n ${APISERVER_PID-} ]] && kill ${APISERVER_PID} 1>&2 2>/dev/null
  unset APISERVER_PID
}

kube::apiserver::cleanup() {
  kube::apiserver::kill

  kube::etcd::cleanup

  kube::log::status "Clean up complete"
}