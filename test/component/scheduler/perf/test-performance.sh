#!/usr/bin/env bash

# Copyright 2014 The Kubernetes Authors All rights reserved.
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

pushd "../../../.."
source "./hack/lib/util.sh"
source "./cluster/lib/logging.sh"
source "./hack/lib/etcd.sh"
popd

cleanup() {
  kube::etcd::cleanup
  kube::log::status "performance test cleanup complete"
}

trap cleanup EXIT

kube::etcd::start
kube::log::status "performance test start"

# TODO: set log-dir and prof output dir.
DIR_BASENAME=$(basename `pwd`)
go test -c -o "${DIR_BASENAME}.test"
# We are using the benchmark suite to do profiling. Because it only runs a few pods and
# theoretically it has less variance.
"./${DIR_BASENAME}.test" -test.bench=. -test.run=xxxx -test.cpuprofile=prof.out -logtostderr=false
kube::log::status "benchmark tests finished"
# Running density tests. It might take a long time.
"./${DIR_BASENAME}.test" -test.run=. -test.timeout=60m
kube::log::status "density tests finished"
