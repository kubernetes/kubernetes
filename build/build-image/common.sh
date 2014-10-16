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

# This script sets up a go workspace locally and builds all go components.
# You can 'source' this file if you want to set up GOPATH in your local shell.

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
cd "${KUBE_ROOT}"

readonly KUBE_TARGET="${KUBE_ROOT}/_output/build"
readonly KUBE_GO_PACKAGE=github.com/GoogleCloudPlatform/kubernetes

server_targets=(
  cmd/proxy
  cmd/apiserver
  cmd/controller-manager
  cmd/kubelet
  plugin/cmd/scheduler
)

client_targets=(
  cmd/kubecfg
  cmd/kubectl
)

mkdir -p "${KUBE_TARGET}"

if [[ ! -f "/kube-build-image" ]]; then
  echo "WARNING: This script should be run in the kube-build container image!" >&2
fi

if [[ -f "/kube-version-defs" ]]; then
  source "/kube-version-defs"
else
  echo "WARNING: No version information provided in build image"
fi

function kube::build::make_binary() {
  local -r gopkg=$1
  local -r bin=${gopkg##*/}

  echo "+++ Building ${bin} for ${GOOS}/${GOARCH}"
  pushd "${KUBE_ROOT}" >/dev/null
  godep go build -ldflags "${KUBE_LD_FLAGS-}" -o "${ARCH_TARGET}/${bin}" "${gopkg}"
  popd >/dev/null
}

function kube::build::make_binaries() {
  [[ $# -gt 0 ]] || {
    echo "!!! Internal error. kube::build::make_binaries called with no targets."
  }

  local -a targets=("$@")
  local -a binaries=()
  local target
  for target in "${targets[@]}"; do
    binaries+=("${KUBE_GO_PACKAGE}/${target}")
  done

  ARCH_TARGET="${KUBE_TARGET}/${GOOS}/${GOARCH}"
  mkdir -p "${ARCH_TARGET}"

  local b
  for b in "${binaries[@]}"; do
    kube::build::make_binary "$b"
  done
}
