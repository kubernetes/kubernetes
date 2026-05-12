#!/usr/bin/env bash

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

TEST_ARGS=""
RUN_PATTERN=".*"
PROFILE_OPTS=""

function usage() {
  echo "usage: $0 <options>"
  echo " -h display this help message"
  echo " -d enable debug logs in tests"
  echo " -r <pattern> regex pattern to match for tests"
  echo " -o <filename> file to write JSON formatted results to"
  echo " -p <id> enable cpu and memory profiles, output written to mem-<id>.out and cpu-<id>.out"
  echo " -c enable custom test configuration"
  echo " -a <name> allocator name, one of RangeAllocator, CloudAllocator, IPAMFromCluster, IPAMFromCloud"
  echo " -k <num> api server qps for allocator"
  echo " -n <num> number of nodes to simulate"
  echo " -m <num> api server qps for node creation"
  echo " -l <num> gce cloud endpoint qps"
  exit 1
}

while getopts ":hdr:o:p:ca:k:n:m:l:" opt; do
  case ${opt} in
    d) TEST_ARGS="${TEST_ARGS} -v=6"
      ;;
    r) RUN_PATTERN="${OPTARG}"
      ;;
    o) TEST_ARGS="${TEST_ARGS} -log ${OPTARG}"
      ;;
    p) PROFILE_OPTS="-memprofile mem-${OPTARG}.out -cpuprofile cpu-${OPTARG}.out"
      ;;
    c) TEST_ARGS="${TEST_ARGS} -custom"
      ;;
    a) TEST_ARGS="${TEST_ARGS} -allocator ${OPTARG}"
      ;;
    k) TEST_ARGS="${TEST_ARGS} -kube-qps ${OPTARG}"
      ;;
    n) TEST_ARGS="${TEST_ARGS} -num-nodes ${OPTARG}"
      ;;
    m) TEST_ARGS="${TEST_ARGS} -create-qps ${OPTARG}"
      ;;
    l) TEST_ARGS="${TEST_ARGS} -cloud-qps ${OPTARG}"
      ;;
    h) usage
      ;;
    \?) usage
      ;;
  esac
done

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/../../../
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::golang::setup_env

DIR_BASENAME=$(dirname "${BASH_SOURCE[0]}")
pushd "${DIR_BASENAME}"

cleanup() {
  popd 2> /dev/null
  kube::etcd::cleanup
  kube::log::status "performance test cleanup complete"
}

trap cleanup EXIT

kube::etcd::start

# Running IPAM tests. It might take a long time.
kube::log::status "performance test (IPAM) start"
go test "${PROFILE_OPTS}" -test.run="${RUN_PATTERN}" -test.timeout=60m -test.short=false -v -args "${TEST_ARGS}"
kube::log::status "... IPAM tests finished."
