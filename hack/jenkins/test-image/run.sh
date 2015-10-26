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

export PATH=${GOPATH}/bin:${PWD}/third_party/etcd:/usr/local/go/bin:${PATH}

./hack/install-etcd.sh
go get -u github.com/jstemmer/go-junit-report
go get golang.org/x/tools/cmd/cover
go get github.com/mattn/goveralls
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
export KUBE_GOVERALLS_BIN="${GOPATH}/bin/goveralls"
export KUBE_TIMEOUT='-timeout 300s'
export KUBE_COVERPROCS=8
export KUBE_INTEGRATION_TEST_MAX_CONCURRENCY=4
export LOG_LEVEL=4

./hack/verify-gofmt.sh
./hack/verify-boilerplate.sh
./hack/verify-description.sh
./hack/verify-flags-underscore.py
./hack/verify-generated-conversions.sh
./hack/verify-generated-deep-copies.sh
./hack/verify-generated-docs.sh
./hack/verify-generated-swagger-docs.sh
./hack/verify-swagger-spec.sh
./hack/verify-linkcheck.sh

./hack/test-go.sh -- -p=2
./hack/test-cmd.sh
./hack/test-integration.sh
./hack/test-update-storage-objects.sh


