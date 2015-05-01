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

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..

function down-clusters {
  for count in $(seq 1 ${clusters}); do
    export KUBE_GCE_INSTANCE_PREFIX=e2e-test-${USER}-${count}
    local cluster_dir=${KUBE_ROOT}/_output/e2e/${KUBE_GCE_INSTANCE_PREFIX}
    export KUBECONFIG=${cluster_dir}/.kubeconfig
    go run ${KUBE_ROOT}/hack/e2e.go -down -v &
  done

  wait
}

function up-clusters {
  for count in $(seq 1 ${clusters}); do
    export KUBE_GCE_INSTANCE_PREFIX=e2e-test-${USER}-${count}
    export KUBE_GCE_CLUSTER_CLASS_B="10.$((${count}*2-1))"
    export MASTER_IP_RANGE="10.$((${count}*2)).0.0/24"

    local cluster_dir=${KUBE_ROOT}/_output/e2e/${KUBE_GCE_INSTANCE_PREFIX}
    mkdir -p ${cluster_dir}
    export KUBECONFIG=${cluster_dir}/.kubeconfig
    go run hack/e2e.go -up -v 2>&1 | tee ${cluster_dir}/up.log &
  done

  fail=0
  for job in $(jobs -p); do
    wait "${job}" || fail=$((fail + 1))
  done

  if (( fail != 0 )); then
    echo "${fail} cluster creation failures. Not continuing with tests."
    exit 1
  fi
}

function run-tests {
  for count in $(seq 1 ${clusters}); do
    export KUBE_GCE_INSTANCE_PREFIX=e2e-test-${USER}-${count}

    local cluster_dir=${KUBE_ROOT}/_output/e2e/${KUBE_GCE_INSTANCE_PREFIX}
    export KUBECONFIG=${cluster_dir}/.kubeconfig
    export E2E_REPORT_DIR=${cluster_dir}
    go run hack/e2e.go -test --test_args="--ginkgo.noColor" "${@:-}" -down 2>&1 | tee ${cluster_dir}/e2e.log &
  done

  wait
}

# Outputs something like:
# _output/e2e/e2e-test-zml-5/junit.xml
#   FAIL: Shell tests that services.sh passes
function post-process {
  echo $1
  cat $1 | python -c '
import sys
from xml.dom.minidom import parse

failed = False
for testcase in parse(sys.stdin).getElementsByTagName("testcase"):
  if len(testcase.getElementsByTagName("failure")) != 0:
    failed = True
    print "  FAIL: {test}".format(test = testcase.getAttribute("name"))
if not failed:
  print "  SUCCESS!"
'
}

function print-results {
  for count in $(seq 1 ${clusters}); do
    for junit in ${KUBE_ROOT}/_output/e2e/e2e-test-${USER}-${count}/junit*.xml; do
      post-process ${junit}
    done
  done
}

if [[ ${KUBERNETES_PROVIDER:-gce} != "gce" ]]; then
  echo "$0 not supported on ${KUBERNETES_PROVIDER} yet" >&2
  exit 1
fi

readonly clusters=${1:-}

if ! [[ "${clusters}" =~ ^[0-9]+$ ]]; then
  echo "Usage: ${0} <number of clusters> [options to hack/e2e.go]" >&2
  exit 1
fi

shift 1

rm -rf _output/e2e
down-clusters
up-clusters
run-tests "${@:-}"
print-results
