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

# This script is intended to be run from kubekins-test container with a
# kubernetes repo mapped in. See k8s.io/test-infra/scenarios/kubernetes_verify.py

export PATH=${GOPATH}/bin:${PWD}/third_party/etcd:/usr/local/go/bin:${PATH}

# Set artifacts directory
export ARTIFACTS_DIR=${WORKSPACE}/artifacts

# Set log level
export LOG_LEVEL=4

# Ensure we are in the k/k directory
cd /go/src/k8s.io/kubernetes

# Pre-load godeps. Retry twice to minimize flakes.
mkdir -p /go/src/k8s.io/kubernetes/_gopath
GOPATH="/go/src/k8s.io/kubernetes/_gopath:${GOPATH}" hack/godep-restore.sh || GOPATH="/go/src/k8s.io/kubernetes/_gopath:${GOPATH}" hack/godep-restore.sh

hack/install-etcd.sh
make verify
