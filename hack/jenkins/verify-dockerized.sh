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
set -o xtrace

retry() {
  for i in {1..5}; do
    "$@" && return 0 || sleep $i
  done
  "$@"
}

# This script is intended to be run from kubekins-test container with a
# kubernetes repo mapped in. See hack/jenkins/gotest-dockerized.sh

export PATH=${GOPATH}/bin:${PWD}/third_party/etcd:/usr/local/go/bin:${PATH}

retry go get github.com/tools/godep && godep version

export LOG_LEVEL=4

cd /go/src/k8s.io/kubernetes

# hack/verify-client-go.sh requires all dependencies exist in the GOPATH.
# the retry helps avoid flakes while keeping total time bounded.
./hack/godep-restore.sh || ./hack/godep-restore.sh

./hack/install-etcd.sh
make verify
