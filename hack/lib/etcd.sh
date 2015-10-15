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

# A set of helpers for starting/running etcd for tests

# Sets ETCD_HOST, ETCD_PORT, and ETCD_PEER_PORT if not already set.
# Also sets ETCD_PID and ETCD_DIR and installs an EXIT trap for cleanup.
kube::etcd::start() {
  ETCD_HOST=${ETCD_HOST:-127.0.0.1}

  if [[ -n "${ETCD_PID-}" ]]; then
    kube::log::error "etcd already started with pid ${ETCD_PID}!"
    return 2
  fi

  which etcd >/dev/null || {
    kube::log::usage "etcd must be in your PATH"
    return 2
  }

  version=$(etcd -version | cut -d " " -f 3)
  if [[ "${version}" < "2.0.0" ]]; then
   kube::log::usage "etcd version 2.0.0 or greater required."
   return 2
  fi

  local -r etcd_start_attempts=3
  for etcd_attempt in $(seq ${etcd_start_attempts}); do
    local client_port=${ETCD_PORT:-$(kube::util::get_random_port)}
    local peer_port=${ETCD_PEER_PORT:-$(kube::util::get_random_port)}
    # Start etcd
    ETCD_DIR=$(mktemp -d 2>/dev/null || mktemp -d -t test-etcd.XXXXXX)
    kube::log::info "Starting etcd with data-dir ${ETCD_DIR}, addr ${ETCD_HOST}:${client_port}, peer addr ${ETCD_HOST}:${peer_port} (attempt ${etcd_attempt}/${etcd_start_attempts})"
    etcd -data-dir ${ETCD_DIR} --bind-addr ${ETCD_HOST}:${client_port} --peer-bind-addr ${ETCD_HOST}:${peer_port} >/dev/null 2>/dev/null &
    ETCD_PID=$!

    echo "Waiting for etcd to come up."
    if kube::util::wait_for_url "http://${ETCD_HOST}:${client_port}/v2/machines" "etcd: " 0.25 80; then
      break
    fi
    # Failed to start up: clean up and try again.
    kube::etcd::cleanup
    if [[ ${etcd_attempt} -ge 3 ]]; then
      kube::log::error "etcd failed to start in ${etcd_attempt} attempts"
      return 1
    fi
  done
  kube::util::trap_add kube::etcd::cleanup EXIT
  ETCD_PORT=${client_port}
  ETCD_PEER_PORT=${peer_port}
  curl -fs -X PUT "http://${ETCD_HOST}:${ETCD_PORT}/v2/keys/_test"
}

kube::etcd::stop() {
  if [[ -n "${ETCD_PID-}" ]]; then
    kill "${ETCD_PID-}" >/dev/null 2>&1 || :
    wait "${ETCD_PID-}" >/dev/null 2>&1 || :
    ETCD_PID=
  fi
}

kube::etcd::clean_etcd_dir() {
  if [[ -n "${ETCD_DIR-}" ]]; then
    rm -rf "${ETCD_DIR-}"
    ETCD_DIR=
  fi
}

kube::etcd::cleanup() {
  kube::etcd::stop
  kube::etcd::clean_etcd_dir
}
