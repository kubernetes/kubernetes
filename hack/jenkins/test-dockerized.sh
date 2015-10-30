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

set -o errexit
set -o nounset
set -o pipefail
set -o xtrace

# Runs the unit and integration tests, producing JUnit-style XML test
# reports in ${WORKSPACE}/artifacts. This script is intended to be run from
# kubekins-test container with a kubernetes repo mapped in. See
# hack/jenkins/gotest-dockerized.sh

export PATH=${GOPATH}/bin:${PWD}/third_party/etcd:/usr/local/go/bin:${PATH}

go get github.com/tools/godep
go get github.com/jstemmer/go-junit-report

# Enable the Go race detector.
export KUBE_RACE=-race
# Disable coverage report
export KUBE_COVER="n"
# Produce a JUnit-style XML test report for Jenkins.
export KUBE_JUNIT_REPORT_DIR=${WORKSPACE}/artifacts
# Save the verbose stdout as well.
export KUBE_KEEP_VERBOSE_TEST_OUTPUT=y
export KUBE_TIMEOUT='-timeout 300s'
export KUBE_INTEGRATION_TEST_MAX_CONCURRENCY=4
export LOG_LEVEL=4
export KUBE_TEST_API_VERSIONS=v1,extensions/v1beta1
export KUBE_TEST_ETCD_PREFIXES=registry

./hack/build-go.sh
godep go install ./...
./hack/travis/install-etcd.sh

./hack/verify-all.sh -v

./hack/test-go.sh -- -p=2
./hack/test-cmd.sh
./hack/test-integration.sh
./hack/test-update-storage-objects.sh


