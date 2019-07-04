#!/usr/bin/env bash

# Copyright 2015 The Kubernetes Authors.
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

# Runs the unit and integration tests, production JUnit-style XML test reports
# in ${WORKSPACE}/_artifacts.

set -o errexit
set -o nounset
set -o pipefail
set -o xtrace

# !!! ALERT !!! Jenkins default $HOME is /var/lib/jenkins, which is
# global across jobs. We change $HOME instead to ${WORKSPACE}, which
# is an incoming variable Jenkins provides us for this job's scratch
# space.
export HOME=${WORKSPACE} # Nothing should want Jenkins $HOME
export GOPATH=${HOME}/_gopath
export PATH=${GOPATH}/bin:${HOME}/third_party/etcd:/usr/local/go/bin:${PATH}

# Install a few things needed by unit and /integration tests.
command -v etcd &>/dev/null || ./hack/install-etcd.sh
go install k8s.io/kubernetes/vendor/github.com/jstemmer/go-junit-report

# Enable the Go race detector.
export KUBE_RACE=-race
# Set artifacts directory
export ARTIFACTS=${ARTIFACTS:-"${WORKSPACE}/artifacts"}
# Produce a JUnit-style XML test report
export KUBE_JUNIT_REPORT_DIR="${ARTIFACTS}"
# Save the verbose stdout as well.
export KUBE_KEEP_VERBOSE_TEST_OUTPUT=y

make test
make test-integration
