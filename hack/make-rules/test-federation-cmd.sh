#!/bin/bash

# Copyright 2016 The Kubernetes Authors.
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

# This script checks that kubectl commands work with federation apiserver.
# This is very similiar to hack/make-rules/test-cmd.sh. The only difference
# being that this tests kubectl with federation-apiserver while
# hack/make-rules/test-cmd.sh tests kubectl with kube-apiserver.
# It does not require Docker.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
source "${KUBE_ROOT}/hack/lib/init.sh"
source "${KUBE_ROOT}/hack/lib/test.sh"

ETCD_HOST=${ETCD_HOST:-127.0.0.1}
ETCD_PORT=${ETCD_PORT:-2399}
API_HOST=${API_HOST:-127.0.0.1}
API_PORT=${API_PORT:-8282}
kube_flags=(
  -s "${API_HOST}:${API_PORT}"
  --match-server-version
)

function cleanup()
{
  [[ -n "${APISERVER_PID-}" ]] && kill "${APISERVER_PID}" 1>&2 2>/dev/null

  kube::etcd::cleanup
  rm -rf "${KUBE_TEMP}"

  kube::log::status "Clean up complete"
}

kube::util::trap_add cleanup EXIT SIGINT
kube::util::ensure-temp-dir

# ensure ~/.kube/config isn't loaded by tests
HOME="${KUBE_TEMP}"

BINS=(
	cmd/kubectl
	federation/cmd/federation-apiserver
)
make -C "${KUBE_ROOT}" WHAT="${BINS[*]}"

kube::etcd::start

# Check kubectl
kube::log::status "Running kubectl with no options"
"${KUBE_OUTPUT_HOSTBIN}/kubectl"

# Start kube-apiserver
kube::log::status "Starting federation-apiserver"

# Admission Controllers to invoke prior to persisting objects in cluster
ADMISSION_CONTROL="NamespaceLifecycle"

"${KUBE_OUTPUT_HOSTBIN}/federation-apiserver" \
  --address="127.0.0.1" \
  --public-address-override="127.0.0.1" \
  --port="${API_PORT}" \
  --admission-control="${ADMISSION_CONTROL}" \
  --etcd-servers="http://${ETCD_HOST}:${ETCD_PORT}" \
  --public-address-override="127.0.0.1" \
  --storage-media-type="${KUBE_TEST_API_STORAGE_TYPE-}" \
  --cert-dir="${TMPDIR:-/tmp/}" \
  --service-cluster-ip-range="10.0.0.0/24" 1>&2 &
APISERVER_PID=$!

kube::util::wait_for_url "http://127.0.0.1:${API_PORT}/" "apiserver"

# Expose kubectl directly for readability
PATH="${KUBE_OUTPUT_HOSTBIN}":$PATH

kube::log::status "Checking kubectl version"
#kubectl version "${kube_flags[@]}"

# TODO: we need to note down the current default namespace and set back to this
# namespace after the tests are done.
kubectl config view
CONTEXT="fed-test"
kubectl config set-context "${CONTEXT}"
kubectl config use-context "${CONTEXT}"

# Tests the basic kubectl commands (get, create, delete, describe) for the given resource.
testResource() {
  resource=$1
  describe=$2

  kube::log::status "Testing ${resource}"

  id_field=".metadata.name"
  # Pre-condition: no resource of this type exists.
  kube::test::get_object_assert "${resource}" "{{range.items}}{{$id_field}}:{{end}}" ''

  kubectl create -f hack/testdata/${resource}.yaml "${kube_flags[@]}"
  kube::test::get_object_assert "${resource}" "{{range.items}}{{$id_field}}:{{end}}" "my${resource}:"

  if [[ "${describe}" = true ]]; then
    kube::test::describe_resource_assert "${resource}" "Name:"
  fi

  # We delete with cascade=false since we do not have a controller running that
  # will delete the finalizers.
  kubectl delete "${resource}" "my${resource}" "${kube_flags[@]}" --cascade=false
}

runTests() {
  # Passing no arguments to create is an error
  ! kubectl create

  # get /version should work.
  kubectl get "${kube_flags[@]}" --raw /version

  # get for unknownresourcetype should fail.
  kube::log::status "Testing Unknown resource"

  UNKNOWN_RESOURCE_ERROR_FILE="${KUBE_TEMP}/unknown-resource-error"

  ### Non-existent resource type should give a recognizeable error
  kubectl get "${kube_flags[@]}" unknownresourcetype 2>${UNKNOWN_RESOURCE_ERROR_FILE} || true
  if grep -q "the server doesn't have a resource type" "${UNKNOWN_RESOURCE_ERROR_FILE}"; then
    kube::log::status "\"kubectl get unknownresourcetype\" returns error as expected: $(cat ${UNKNOWN_RESOURCE_ERROR_FILE})"
  else
    kube::log::status "\"kubectl get unknownresourcetype\" returns unexpected error or non-error: $(cat ${UNKNOWN_RESOURCE_ERROR_FILE})"
    exit 1
  fi
  rm "${UNKNOWN_RESOURCE_ERROR_FILE}"

  # Test namespaces
  testResource "ns" true

  # Create namespace "default" for testing namespaced resources.
  kubectl create ns "default" "${kube_flags[@]}"

  # Test events.
  # Note: describe is not supported for events.
  testResource "events" false

  # Test secrets.
  testResource "secrets" true

  # Test replicasets.
  # TODO: Test `describe rs` when
  # https://github.com/kubernetes/kubernetes/issues/33309 is fixed.
  testResource "rs" false

  # Test services
  testResource "svc" true

  # Test ingress
  testResource "ingress" true
}

runTests

kube::log::status "TEST PASSED"
