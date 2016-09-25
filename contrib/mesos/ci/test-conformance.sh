#!/bin/bash

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

# Deploys a test cluster, runs the conformance tests, and destroys the test cluster.
#
# Prerequisite:
# ./cluster/mesos/docker/test/build.sh
#
# Example Usage:
# ./contrib/mesos/ci/test-conformance.sh -v=2

set -o errexit
set -o nounset
set -o pipefail
set -o errtrace

TEST_ARGS="$@"

KUBE_ROOT=$(cd "$(dirname "${BASH_SOURCE}")/../../.." && pwd)

TEST_CMD="KUBERNETES_CONFORMANCE_TEST=y KUBECONFIG=~/.kube/config go run hack/e2e.go --test --test_args=\"--ginkgo.focus=\\[Conformance\\]\""
if [ -n "${CONFORMANCE_BRANCH}" ]; then
	# create a CONFORMANCE_BRANCH clone in a subdirectory
	TEST_CMD="
git fetch https://github.com/kubernetes/kubernetes --tags -q ${CONFORMANCE_BRANCH} &&
git branch -f ${CONFORMANCE_BRANCH} FETCH_HEAD &&
git clone -s -b ${CONFORMANCE_BRANCH} . conformance &&
cd conformance && make all && ${TEST_CMD}"
fi

"${KUBE_ROOT}/contrib/mesos/ci/run-with-cluster.sh" ${TEST_CMD} ${TEST_ARGS}
