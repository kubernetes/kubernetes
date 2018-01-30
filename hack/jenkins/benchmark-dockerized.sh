#!/bin/bash

# Copyright 2018 The Kubernetes Authors.
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

# Runs benchmark integration tests, producing JUnit-style XML test
# reports in ${WORKSPACE}/artifacts. This script is intended to be run from
# kubekins-test container with a kubernetes repo mounted (at the path
# /go/src/k8s.io/kubernetes). See k8s.io/test-infra/scenarios/kubernetes_verify.py.

export PATH=${GOPATH}/bin:${PWD}/third_party/etcd:/usr/local/go/bin:${PATH}

retry go get github.com/jstemmer/go-junit-report
retry go get github.com/cespare/prettybench

# Disable the Go race detector.
export KUBE_RACE=" "
# Disable coverage report
export KUBE_COVER="n"
# Produce a JUnit-style XML test report.
export KUBE_JUNIT_REPORT_DIR=${WORKSPACE}/artifacts
export ARTIFACTS_DIR=${WORKSPACE}/artifacts

mkdir -p "${ARTIFACTS_DIR}"
cd /go/src/k8s.io/kubernetes

./hack/install-etcd.sh

# Run the benchmark tests and pretty-print the results into a separate file.
make test-integration WHAT="$*" KUBE_TEST_ARGS="-run='XXX' -bench=. -benchmem" \
  | tee >(prettybench -no-passthrough > ${ARTIFACTS_DIR}/BenchmarkResults.txt)
