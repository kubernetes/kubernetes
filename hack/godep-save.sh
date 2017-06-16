#!/bin/bash

# Copyright 2016 The Kubernetes Authors.
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

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"
source "${KUBE_ROOT}/hack/lib/util.sh"

kube::util::ensure_godep_version v79

# Some things we want in godeps aren't code dependencies, so ./...
# won't pick them up.
REQUIRED_BINS=(
  "github.com/ugorji/go/codec/codecgen"
  "github.com/onsi/ginkgo/ginkgo"
  "github.com/jteeuwen/go-bindata/go-bindata"
  "./..."
)

pushd "${KUBE_ROOT}" > /dev/null
  GOPATH=${GOPATH}:${KUBE_ROOT}/staging godep save "${REQUIRED_BINS[@]}"

  # create a symlink in vendor directory pointing to the staging client. This
  # let other packages use the staging client as if it were vendored.
  if [ ! -e "vendor/k8s.io/client-go" ]; then
    ln -s ../../staging/src/k8s.io/client-go vendor/k8s.io/client-go
  fi
  if [ ! -e "vendor/k8s.io/apiserver" ]; then
    ln -s ../../staging/src/k8s.io/apiserver vendor/k8s.io/apiserver
  fi
  if [ ! -e "vendor/k8s.io/apimachinery" ]; then
    ln -s ../../staging/src/k8s.io/apimachinery vendor/k8s.io/apimachinery
  fi
  if [ ! -e "vendor/k8s.io/kube-aggregator" ]; then
    ln -s ../../staging/src/k8s.io/kube-aggregator vendor/k8s.io/kube-aggregator
  fi
  if [ ! -e "vendor/k8s.io/apiextensions-apiserver" ]; then
    ln -s ../../staging/src/k8s.io/apiextensions-apiserver vendor/k8s.io/apiextensions-apiserver
  fi
  if [ ! -e "vendor/k8s.io/sample-apiserver" ]; then
    ln -s ../../staging/src/k8s.io/sample-apiserver vendor/k8s.io/sample-apiserver
  fi
  if [ ! -e "vendor/k8s.io/metrics" ]; then
    ln -s ../../staging/src/k8s.io/metrics vendor/k8s.io/metrics
  fi
popd > /dev/null

echo "Don't forget to run hack/update-godep-licenses.sh if you added or removed a dependency!"
