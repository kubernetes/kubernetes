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

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/../..
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::golang::setup_env
kube::golang::setup_gomaxprocs
kube::util::require-jq

# start the cache mutation detector by default so that cache mutators will be found
KUBE_CACHE_MUTATION_DETECTOR="${KUBE_CACHE_MUTATION_DETECTOR:-true}"
export KUBE_CACHE_MUTATION_DETECTOR

# panic the server on watch decode errors since they are considered coder mistakes
KUBE_PANIC_WATCH_DECODE_ERROR="${KUBE_PANIC_WATCH_DECODE_ERROR:-true}"
export KUBE_PANIC_WATCH_DECODE_ERROR

KUBE_INTEGRATION_TEST_MAX_CONCURRENCY=${KUBE_INTEGRATION_TEST_MAX_CONCURRENCY:-"-1"}
if [[ ${KUBE_INTEGRATION_TEST_MAX_CONCURRENCY} -gt 0 ]]; then
  GOMAXPROCS=${KUBE_INTEGRATION_TEST_MAX_CONCURRENCY}
  export GOMAXPROCS
  kube::log::status "Setting parallelism to ${GOMAXPROCS}"
fi

# Give integration tests longer to run by default.
KUBE_TIMEOUT=${KUBE_TIMEOUT:--timeout=600s}
LOG_LEVEL=${LOG_LEVEL:-2}
KUBE_TEST_ARGS=${KUBE_TEST_ARGS:-}
# Default glog module settings.
KUBE_TEST_VMODULE=${KUBE_TEST_VMODULE:-""}

kube::test::find_integration_test_pkgs() {
  (
    cd "${KUBE_ROOT}"

    # Get a list of all the modules in this workspace.
    local -a workspace_module_patterns
    kube::util::read-array workspace_module_patterns < <(
        go list -m -json | jq -r '.Dir' \
        | while read -r D; do
            SUB="${D}/test/integration";
            test -d "${SUB}" && echo "${SUB}/...";
        done)

    # Get a list of all packages which have test files.
    go list -find \
        -f '{{if or (gt (len .TestGoFiles) 0) (gt (len .XTestGoFiles) 0)}}{{.ImportPath}}{{end}}' \
        "${workspace_module_patterns[@]}"
  )
}

CLEANUP_REQUIRED=
cleanup() {
  if [[ -z "${CLEANUP_REQUIRED}" ]]; then
    return
  fi
  kube::log::status "Cleaning up etcd"
  kube::etcd::cleanup
  CLEANUP_REQUIRED=
  kube::log::status "Integration test cleanup complete"
}

runTests() {
  kube::log::status "Starting etcd instance"
  CLEANUP_REQUIRED=1
  kube::etcd::start
  # shellcheck disable=SC2034
  local ETCD_SCRAPE_PID # Set in kube::etcd::start_scraping, used in cleanup
  kube::etcd::start_scraping
  kube::log::status "Running integration test cases"

  # shellcheck disable=SC2034
  # KUBE_RACE and MAKEFLAGS are used in the downstream make, and we set them to
  # empty here to ensure that we aren't unintentionally consuming them from the
  # previous make invocation.
  KUBE_TEST_ARGS="${SHORT:--short=true} --vmodule=${KUBE_TEST_VMODULE} ${KUBE_TEST_ARGS}" \
      WHAT="${WHAT:-$(kube::test::find_integration_test_pkgs | paste -sd' ' -)}" \
      GOFLAGS="${GOFLAGS:-}" \
      KUBE_TIMEOUT="${KUBE_TIMEOUT}" \
      KUBE_RACE=${KUBE_RACE:-""} \
      MAKEFLAGS="" \
      make -C "${KUBE_ROOT}" test

  cleanup
}

checkEtcdOnPath() {
  kube::log::status "Checking etcd is on PATH"
  which etcd && return
  kube::log::status "Cannot find etcd, cannot run integration tests."
  kube::log::status "Please see https://git.k8s.io/community/contributors/devel/sig-testing/integration-tests.md#install-etcd-dependency for instructions."
  kube::log::usage "You can use 'hack/install-etcd.sh' to install a copy in third_party/."
  return 1
}

checkEtcdOnPath

# Run cleanup to stop etcd on interrupt or other kill signal.
trap cleanup EXIT

runTests
