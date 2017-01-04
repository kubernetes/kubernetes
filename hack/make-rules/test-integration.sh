#!/bin/bash

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

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
source "${KUBE_ROOT}/hack/lib/init.sh"
# Lists of API Versions of each groups that should be tested, groups are
# separated by comma, lists are separated by semicolon. e.g.,
# "v1,compute/v1alpha1,experimental/v1alpha2;v1,compute/v2,experimental/v1alpha3"
# TODO: It's going to be:
# KUBE_TEST_API_VERSIONS=${KUBE_TEST_API_VERSIONS:-"v1,extensions/v1beta1"}
# FIXME: due to current implementation of a test client (see: pkg/api/testapi/testapi.go)
# ONLY the last version is tested in each group.
ALL_VERSIONS_CSV=$(IFS=',';echo "${KUBE_AVAILABLE_GROUP_VERSIONS[*]// /,}";IFS=$)
KUBE_TEST_API_VERSIONS="${KUBE_TEST_API_VERSIONS:-${ALL_VERSIONS_CSV}}"

# Give integration tests longer to run
# TODO: allow a larger value to be passed in
#KUBE_TIMEOUT=${KUBE_TIMEOUT:--timeout 240s}
KUBE_TIMEOUT="-timeout 600s"
KUBE_INTEGRATION_TEST_MAX_CONCURRENCY=${KUBE_INTEGRATION_TEST_MAX_CONCURRENCY:-"-1"}
LOG_LEVEL=${LOG_LEVEL:-2}
KUBE_TEST_ARGS=${KUBE_TEST_ARGS:-}

kube::test::find_integration_test_dirs() {
  (
    cd ${KUBE_ROOT}
    find test/integration/${1-} -name '*_test.go' -print0 \
      | xargs -0n1 dirname | sed "s|^|${KUBE_GO_PACKAGE}/|" \
      | LC_ALL=C sort -u
  )
}

cleanup() {
  kube::log::status "Cleaning up etcd"
  kube::etcd::cleanup
  kube::log::status "Integration test cleanup complete"
}

runTests() {
  kube::log::status "Starting etcd instance"
  kube::etcd::start
  kube::log::status "Running integration test cases"

  # TODO: Re-enable race detection when we switch to a thread-safe etcd client
  # KUBE_RACE="-race"
  make -C "${KUBE_ROOT}" test \
      WHAT="$(kube::test::find_integration_test_dirs ${2-} | paste -sd' ' -) $(echo ${@:3})" \
      KUBE_GOFLAGS="${KUBE_GOFLAGS:-} -tags 'integration no-docker'" \
      KUBE_TEST_ARGS="${KUBE_TEST_ARGS:-} ${SHORT:--short=true} --vmodule=garbage*collector*=6 --alsologtostderr=true" \
      KUBE_RACE="" \
      KUBE_TIMEOUT="${KUBE_TIMEOUT}" \
      KUBE_TEST_API_VERSIONS="$1"

  cleanup
}

checkEtcdOnPath() {
  kube::log::status "Checking etcd is on PATH"
  which etcd && return
  kube::log::status "Cannot find etcd, cannot run integration tests."
  kube::log::status "Please see docs/devel/testing.md for instructions."
  return 1
}

checkEtcdOnPath

# Run cleanup to stop etcd on interrupt or other kill signal.
trap cleanup EXIT

# If a test case is specified, just run once with v1 API version and exit
if [[ -n "${KUBE_TEST_ARGS}" ]]; then
  runTests v1
fi

# Pass arguments that begin with "-" and move them to goflags.
what_flags=()
for arg in "$@"; do
  if [[ "${arg}" == -* ]]; then
    what_flags+=("${arg}")
  fi
done
if [[ "${#what_flags[@]}" -eq 0 ]]; then
  what_flags=''
fi


# Convert the CSV to an array of API versions to test
IFS=';' read -a apiVersions <<< "${KUBE_TEST_API_VERSIONS}"
for apiVersion in "${apiVersions[@]}"; do
  runTests "${apiVersion}" "${1-}" "${what_flags[@]}"
done
