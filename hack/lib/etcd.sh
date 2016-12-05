#!/bin/bash

# Copyright 2014 The Kubernetes Authors.
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

ETCD_VERSION=${ETCD_VERSION:-3.0.14}
ETCD_HOST=${ETCD_HOST:-127.0.0.1}
ETCD_PORT=${ETCD_PORT:-2379}

kube::etcd::start() {
  which etcd >/dev/null || {
    kube::log::usage "etcd must be in your PATH"
    exit 1
  }

  if pgrep etcd >/dev/null 2>&1; then
    kube::log::usage "etcd appears to already be running on this machine (`pgrep -l etcd`) (or its a zombie and you need to kill its parent)."
    kube::log::usage "retry after you resolve this etcd error."
    exit 1
  fi

  version=$(etcd --version | tail -n +1 | head -n 1 | cut -d " " -f 3)
  if [[ "${version}" < "${ETCD_VERSION}" ]]; then
   kube::log::usage "etcd version ${ETCD_VERSION} or greater required."
   kube::log::info "You can use 'hack/install-etcd.sh' to install a copy in third_party/."
   exit 1
  fi

  # Start etcd
  ETCD_DIR=${ETCD_DIR:-$(mktemp -d 2>/dev/null || mktemp -d -t test-etcd.XXXXXX)}
  if [[ -d "${ARTIFACTS_DIR:-}" ]]; then
    ETCD_LOGFILE="${ARTIFACTS_DIR}/etcd.$(uname -n).$(id -un).log.DEBUG.$(date +%Y%m%d-%H%M%S).$$"
  else
    ETCD_LOGFILE=/dev/null
  fi
  kube::log::info "etcd --advertise-client-urls http://${ETCD_HOST}:${ETCD_PORT} --data-dir ${ETCD_DIR} --listen-client-urls http://${ETCD_HOST}:${ETCD_PORT} --debug > \"${ETCD_LOGFILE}\" 2>/dev/null"
  etcd --advertise-client-urls http://${ETCD_HOST}:${ETCD_PORT} --data-dir ${ETCD_DIR} --listen-client-urls http://${ETCD_HOST}:${ETCD_PORT} --debug 2> "${ETCD_LOGFILE}" >/dev/null &
  ETCD_PID=$!

  echo "Waiting for etcd to come up."
  kube::util::wait_for_url "http://${ETCD_HOST}:${ETCD_PORT}/v2/machines" "etcd: " 0.25 80
  curl -fs -X PUT "http://${ETCD_HOST}:${ETCD_PORT}/v2/keys/_test"
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

kube::etcd::install() {
  (
    cd "${KUBE_ROOT}/third_party"
    if [[ $(uname) == "Darwin" ]]; then
      download_file="etcd-v${ETCD_VERSION}-darwin-amd64.zip"
      url="https://github.com/coreos/etcd/releases/download/v${ETCD_VERSION}/${download_file}"
      kube::util::download_file "${url}" "${download_file}"
      unzip -o "${download_file}"
      ln -fns "etcd-v${ETCD_VERSION}-darwin-amd64" etcd
      rm "${download_file}"
    else
      url="https://github.com/coreos/etcd/releases/download/v${ETCD_VERSION}/etcd-v${ETCD_VERSION}-linux-amd64.tar.gz"
      download_file="etcd-v${ETCD_VERSION}-linux-amd64.tar.gz"
      kube::util::download_file "${url}" "${download_file}"
      tar xzf "${download_file}"
      ln -fns "etcd-v${ETCD_VERSION}-linux-amd64" etcd
    fi
    kube::log::info "etcd v${ETCD_VERSION} installed. To use:"
    kube::log::info "export PATH=$(pwd)/etcd:\${PATH}"
  )
}
