#!/usr/bin/env bash

# Copyright 2014 The Kubernetes Authors.
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

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../../../..
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::golang::setup_env

DIR_BASENAME=$(dirname "${BASH_SOURCE}")
pushd ${DIR_BASENAME}

cleanup() {
  popd 2> /dev/null
  kube::etcd::cleanup
  kube::log::status "performance test cleanup complete"
}

trap cleanup EXIT

kube::etcd::start
kube::log::status "performance test start"

# We are using the benchmark suite to do profiling. Because it only runs a few pods and
# theoretically it has less variance.
if ${RUN_BENCHMARK:-false}; then
  go test -c -o "perf.test"
  "./perf.test" -test.bench=. -test.run=xxxx -test.cpuprofile=prof.out
  kube::log::status "benchmark tests finished"
fi
# Running density tests. It might take a long time.
go test -test.run=. -test.timeout=60m
kube::log::status "density tests finished"
