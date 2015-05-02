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

kube::etcd::start() {
  local host=${ETCD_HOST:-127.0.0.1}
  local port=${ETCD_PORT:-4001}
  local testhost=${ETCD_PUBLIC_HOST:-localhost}

  which etcd >/dev/null || {
    kube::log::usage "etcd must be in your PATH"
    exit 1
  }

  if pgrep etcd >/dev/null 2>&1; then
    kube::log::usage "etcd appears to already be running on this machine. Please kill and restart the test."
    exit 1
  fi

  version=$(etcd -version | cut -d " " -f 3)
  if [[ "${version}" < "2.0.0" ]]; then
   kube::log::usage "etcd version 2.0.0 or greater required."
   exit 1
  fi

  # Start etcd
  ETCD_DIR=$(mktemp -d -t test-etcd.XXXXXX)
  kube::log::usage "etcd -data-dir ${ETCD_DIR} --bind-addr ${host}:${port} >/dev/null 2>/dev/null"
  etcd -data-dir ${ETCD_DIR} --bind-addr ${host}:${port} >/dev/null 2>/dev/null &
  ETCD_PID=$!

  echo "Waiting for etcd to come up."
  kube::util::wait_for_url "http://${host}:${port}/v2/machines" "etcd: " 0.25 80
  curl -X PUT "http://${host}:${port}/v2/keys/_test"
}

kube::etcd::stop() {
  kill "${ETCD_PID-}" >/dev/null 2>&1 || :
  wait "${ETCD_PID-}" >/dev/null 2>&1 || :
}

kube::etcd::clean_etcd_dir() {
  rm -rf "${ETCD_DIR-}"
}

kube::etcd::cleanup() {
  kube::etcd::stop
  kube::etcd::clean_etcd_dir
}
