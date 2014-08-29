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

cd $(dirname "${BASH_SOURCE}")/../.. >/dev/null
readonly KUBE_REPO_ROOT="${PWD}"
readonly KUBE_TARGET="${KUBE_REPO_ROOT}/_output/build"
readonly KUBE_GO_PACKAGE=github.com/GoogleCloudPlatform/kubernetes

mkdir -p "${KUBE_TARGET}"

if [[ ! -f "/kube-build-image" ]]; then
  echo "WARNING: This script should be run in the kube-build conrtainer image!" >&2
fi

function make-binaries() {
  readonly BINARIES="
    proxy
    integration
    apiserver
    controller-manager
    kubelet
    kubecfg"

  ARCH_TARGET="${KUBE_TARGET}/${GOOS}/${GOARCH}"
  mkdir -p "${ARCH_TARGET}"

  function make-binary() {
    echo "+++ Building $1 for ${GOOS}/${GOARCH}"
    godep go build \
      -o "${ARCH_TARGET}/$1" \
      github.com/GoogleCloudPlatform/kubernetes/cmd/$1
  }

  if [[ -n $1 ]]; then
    make-binary $1
    exit 0
  fi

  for b in ${BINARIES}; do
    make-binary $b
  done
}
