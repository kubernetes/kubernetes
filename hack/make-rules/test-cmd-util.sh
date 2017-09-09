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

# This contains util code for testing kubectl.

set -o errexit
set -o nounset
set -o pipefail

# Set locale to ensure english responses from kubectl commands
export LANG=C

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
# Expects the following has already been done by whatever sources this script
# source "${KUBE_ROOT}/hack/lib/init.sh"
# source "${KUBE_ROOT}/hack/lib/test.sh"

ETCD_HOST=${ETCD_HOST:-127.0.0.1}
ETCD_PORT=${ETCD_PORT:-2379}
API_PORT=${API_PORT:-8080}
SECURE_API_PORT=${SECURE_API_PORT:-6443}
API_HOST=${API_HOST:-127.0.0.1}
KUBELET_HEALTHZ_PORT=${KUBELET_HEALTHZ_PORT:-10248}
CTLRMGR_PORT=${CTLRMGR_PORT:-10252}
PROXY_HOST=127.0.0.1 # kubectl only serves on localhost.

IMAGE_NGINX="gcr.io/google-containers/nginx:1.7.9"
IMAGE_DEPLOYMENT_R1="gcr.io/google-containers/nginx:test-cmd"  # deployment-revision1.yaml
IMAGE_DEPLOYMENT_R2="$IMAGE_NGINX"  # deployment-revision2.yaml
IMAGE_PERL="gcr.io/google-containers/perl"
IMAGE_PAUSE_V2="gcr.io/google-containers/pause:2.0"
IMAGE_DAEMONSET_R2="gcr.io/google-containers/pause:latest"
IMAGE_DAEMONSET_R2_2="gcr.io/google-containers/nginx:test-cmd"  # rollingupdate-daemonset-rv2.yaml
IMAGE_STATEFULSET_R1="gcr.io/google_containers/nginx-slim:0.7"
IMAGE_STATEFULSET_R2="gcr.io/google_containers/nginx-slim:0.8"

# Expose kubectl directly for readability
PATH="${KUBE_OUTPUT_HOSTBIN}":$PATH

# Define variables for resource types to prevent typos.
clusterroles="clusterroles"
configmaps="configmaps"
csr="csr"
deployments="deployments"
horizontalpodautoscalers="horizontalpodautoscalers"
metrics="metrics"
namespaces="namespaces"
nodes="nodes"
persistentvolumeclaims="persistentvolumeclaims"
persistentvolumes="persistentvolumes"
pods="pods"
podtemplates="podtemplates"
replicasets="replicasets"
replicationcontrollers="replicationcontrollers"
roles="roles"
secrets="secrets"
serviceaccounts="serviceaccounts"
services="services"
statefulsets="statefulsets"
static="static"
storageclass="storageclass"
subjectaccessreviews="subjectaccessreviews"
selfsubjectaccessreviews="selfsubjectaccessreviews"
thirdpartyresources="thirdpartyresources"
customresourcedefinitions="customresourcedefinitions"
daemonsets="daemonsets"
controllerrevisions="controllerrevisions"


# include shell2junit library
sh2ju="${KUBE_ROOT}/third_party/forked/shell2junit/sh2ju.sh"
if [[ -f "${sh2ju}" ]]; then
  source "${sh2ju}"
else
  echo "failed to find third_party/forked/shell2junit/sh2ju.sh"
  exit 1
fi

# record_command runs the command and records its output/error messages in junit format
# it expects the first to be the name of the command
# Example:
# record_command run_kubectl_tests
#
# WARNING: Variable changes in the command will NOT be effective after record_command returns.
#          This is because the command runs in subshell.
function record_command() {
    set +o nounset
    set +o errexit

    local name="$1"
    local output="${KUBE_JUNIT_REPORT_DIR:-/tmp/junit-results}"
    echo "Recording: ${name}"
    echo "Running command: $@"
    juLog -output="${output}" -class="test-cmd" -name="${name}" "$@"
    if [[ $? -ne 0 ]]; then
      echo "Error when running ${name}"
      foundError="True"
    fi

    set -o nounset
    set -o errexit
}

# Stops the running kubectl proxy, if there is one.
function stop-proxy()
{
  [[ -n "${PROXY_PORT-}" ]] && kube::log::status "Stopping proxy on port ${PROXY_PORT}"
  [[ -n "${PROXY_PID-}" ]] && kill "${PROXY_PID}" 1>&2 2>/dev/null
  [[ -n "${PROXY_PORT_FILE-}" ]] && rm -f ${PROXY_PORT_FILE}
  PROXY_PID=
  PROXY_PORT=
  PROXY_PORT_FILE=
}

# Starts "kubect proxy" to test the client proxy. $1: api_prefix
function start-proxy()
{
  stop-proxy

  PROXY_PORT_FILE=$(mktemp proxy-port.out.XXXXX)
  kube::log::status "Starting kubectl proxy on random port; output file in ${PROXY_PORT_FILE}; args: ${1-}"


  if [ $# -eq 0 ]; then
    kubectl proxy --port=0 --www=. 1>${PROXY_PORT_FILE} 2>&1 &
  else
    kubectl proxy --port=0 --www=. --api-prefix="$1" 1>${PROXY_PORT_FILE} 2>&1 &
  fi
  PROXY_PID=$!
  PROXY_PORT=

  local attempts=0
  while [[ -z ${PROXY_PORT} ]]; do
    if (( ${attempts} > 9 )); then
      kill "${PROXY_PID}"
      kube::log::error_exit "Couldn't start proxy. Failed to read port after ${attempts} tries. Got: $(cat ${PROXY_PORT_FILE})"
    fi
    sleep .5
    kube::log::status "Attempt ${attempts} to read ${PROXY_PORT_FILE}..."
    PROXY_PORT=$(sed 's/.*Starting to serve on 127.0.0.1:\([0-9]*\)$/\1/'< ${PROXY_PORT_FILE})
    attempts=$((attempts+1))
  done

  kube::log::status "kubectl proxy running on port ${PROXY_PORT}"

  # We try checking kubectl proxy 30 times with 1s delays to avoid occasional
  # failures.
  if [ $# -eq 0 ]; then
    kube::util::wait_for_url "http://127.0.0.1:${PROXY_PORT}/healthz" "kubectl proxy"
  else
    kube::util::wait_for_url "http://127.0.0.1:${PROXY_PORT}/$1/healthz" "kubectl proxy --api-prefix=$1"
  fi
}

function cleanup()
{
  [[ -n "${APISERVER_PID-}" ]] && kill "${APISERVER_PID}" 1>&2 2>/dev/null
  [[ -n "${CTLRMGR_PID-}" ]] && kill "${CTLRMGR_PID}" 1>&2 2>/dev/null
  [[ -n "${KUBELET_PID-}" ]] && kill "${KUBELET_PID}" 1>&2 2>/dev/null
  stop-proxy

  kube::etcd::cleanup
  rm -rf "${KUBE_TEMP}"

  local junit_dir="${KUBE_JUNIT_REPORT_DIR:-/tmp/junit-results}"
  echo "junit report dir:" ${junit_dir}

  kube::log::status "Clean up complete"
}

# Executes curl against the proxy. $1 is the path to use, $2 is the desired
# return code. Prints a helpful message on failure.
function check-curl-proxy-code()
{
  local status
  local -r address=$1
  local -r desired=$2
  local -r full_address="${PROXY_HOST}:${PROXY_PORT}${address}"
  status=$(curl -w "%{http_code}" --silent --output /dev/null "${full_address}")
  if [ "${status}" == "${desired}" ]; then
    return 0
  fi
  echo "For address ${full_address}, got ${status} but wanted ${desired}"
  return 1
}

# TODO: Remove this function when we do the retry inside the kubectl commands. See #15333.
function kubectl-with-retry()
{
  ERROR_FILE="${KUBE_TEMP}/kubectl-error"
  preserve_err_file=${PRESERVE_ERR_FILE-false}
  for count in {0..3}; do
    kubectl "$@" 2> ${ERROR_FILE} || true
    if grep -q "the object has been modified" "${ERROR_FILE}"; then
      kube::log::status "retry $1, error: $(cat ${ERROR_FILE})"
      rm "${ERROR_FILE}"
      sleep $((2**count))
    else
      if [ "$preserve_err_file" != true ] ; then
        rm "${ERROR_FILE}"
      fi
      break
    fi
  done
}

# Waits for the pods with the given label to match the list of names. Don't call
# this function unless you know the exact pod names, or expect no pods.
# $1: label to match
# $2: list of pod names sorted by name
# Example invocation:
# wait-for-pods-with-label "app=foo" "nginx-0nginx-1"
function wait-for-pods-with-label()
{
  local i
  for i in $(seq 1 10); do
    kubeout=`kubectl get po -l $1 --template '{{range.items}}{{.metadata.name}}{{end}}' --sort-by metadata.name "${kube_flags[@]}"`
    if [[ $kubeout = $2 ]]; then
        return
    fi
    echo Waiting for pods: $2, found $kubeout
    sleep $i
  done
  kube::log::error_exit "Timeout waiting for pods with label $1"
}

# Code to be run before running the tests.
setup() {
  kube::util::trap_add cleanup EXIT SIGINT
  kube::util::ensure-temp-dir
  # ensure ~/.kube/config isn't loaded by tests
  HOME="${KUBE_TEMP}"

  kube::etcd::start

  # Find a standard sed instance for use with edit scripts
  SED=sed
  if which gsed &>/dev/null; then
    SED=gsed
  fi
  if ! ($SED --version 2>&1 | grep -q GNU); then
    echo "!!! GNU sed is required.  If on OS X, use 'brew install gnu-sed'."
    exit 1
  fi

  kube::log::status "Building kubectl"
  make -C "${KUBE_ROOT}" WHAT="cmd/kubectl"

  # Check kubectl
  kube::log::status "Running kubectl with no options"
  "${KUBE_OUTPUT_HOSTBIN}/kubectl"

  # TODO: we need to note down the current default namespace and set back to this
  # namespace after the tests are done.
  kubectl config view
  CONTEXT="test"
  kubectl config set-context "${CONTEXT}"
  kubectl config use-context "${CONTEXT}"

  kube::log::status "Setup complete"
}

########################################################
# Kubectl version (--short, --client, --output) #
########################################################
run_kubectl_version_tests() {
  set -o nounset
  set -o errexit

  kube::log::status "Testing kubectl version"
  TEMP="${KUBE_TEMP}"

  kubectl get "${kube_flags[@]}" --raw /version

  # create version files, one for the client, one for the server.
  # these are the files we will use to ensure that the remainder output is correct
  kube::test::version::object_to_file "Client" "" "${TEMP}/client_version_test"
  kube::test::version::object_to_file "Server" "" "${TEMP}/server_version_test"

  kube::log::status "Testing kubectl version: check client only output matches expected output"
  kube::test::version::object_to_file "Client" "--client" "${TEMP}/client_only_version_test"
  kube::test::version::object_to_file "Client" "--client" "${TEMP}/server_client_only_version_test"
  kube::test::version::diff_assert "${TEMP}/client_version_test" "eq" "${TEMP}/client_only_version_test" "the flag '--client' shows correct client info"
  kube::test::version::diff_assert "${TEMP}/server_version_test" "ne" "${TEMP}/server_client_only_version_test" "the flag '--client' correctly has no server version info"

  kube::log::status "Testing kubectl version: verify json output"
  kube::test::version::json_client_server_object_to_file "" "clientVersion" "${TEMP}/client_json_version_test"
  kube::test::version::json_client_server_object_to_file "" "serverVersion" "${TEMP}/server_json_version_test"
  kube::test::version::diff_assert "${TEMP}/client_version_test" "eq" "${TEMP}/client_json_version_test" "--output json has correct client info"
  kube::test::version::diff_assert "${TEMP}/server_version_test" "eq" "${TEMP}/server_json_version_test" "--output json has correct server info"

  kube::log::status "Testing kubectl version: verify json output using additional --client flag does not contain serverVersion"
  kube::test::version::json_client_server_object_to_file "--client" "clientVersion" "${TEMP}/client_only_json_version_test"
  kube::test::version::json_client_server_object_to_file "--client" "serverVersion" "${TEMP}/server_client_only_json_version_test"
  kube::test::version::diff_assert "${TEMP}/client_version_test" "eq" "${TEMP}/client_only_json_version_test" "--client --output json has correct client info"
  kube::test::version::diff_assert "${TEMP}/server_version_test" "ne" "${TEMP}/server_client_only_json_version_test" "--client --output json has no server info"

  kube::log::status "Testing kubectl version: compare json output using additional --short flag"
  kube::test::version::json_client_server_object_to_file "--short" "clientVersion" "${TEMP}/client_short_json_version_test"
  kube::test::version::json_client_server_object_to_file "--short" "serverVersion" "${TEMP}/server_short_json_version_test"
  kube::test::version::diff_assert "${TEMP}/client_version_test" "eq" "${TEMP}/client_short_json_version_test" "--short --output client json info is equal to non short result"
  kube::test::version::diff_assert "${TEMP}/server_version_test" "eq" "${TEMP}/server_short_json_version_test" "--short --output server json info is equal to non short result"

  kube::log::status "Testing kubectl version: compare json output with yaml output"
  kube::test::version::json_object_to_file "" "${TEMP}/client_server_json_version_test"
  kube::test::version::yaml_object_to_file "" "${TEMP}/client_server_yaml_version_test"
  kube::test::version::diff_assert "${TEMP}/client_server_json_version_test" "eq" "${TEMP}/client_server_yaml_version_test" "--output json/yaml has identical information"

  set +o nounset
  set +o errexit
}

# Runs all pod related tests.
run_pod_tests() {
  set -o nounset
  set -o errexit

  kube::log::status "Testing kubectl(v1:pods)"

  ### Create POD valid-pod from JSON
  # Pre-condition: no POD exists
  create_and_use_new_namespace
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create "${kube_flags[@]}" -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml
  # Post-condition: valid-pod POD is created
  kubectl get "${kube_flags[@]}" pods -o json
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'
  kube::test::get_object_assert 'pod valid-pod' "{{$id_field}}" 'valid-pod'
  kube::test::get_object_assert 'pod/valid-pod' "{{$id_field}}" 'valid-pod'
  kube::test::get_object_assert 'pods/valid-pod' "{{$id_field}}" 'valid-pod'
  # Repeat above test using jsonpath template
  kube::test::get_object_jsonpath_assert pods "{.items[*]$id_field}" 'valid-pod'
  kube::test::get_object_jsonpath_assert 'pod valid-pod' "{$id_field}" 'valid-pod'
  kube::test::get_object_jsonpath_assert 'pod/valid-pod' "{$id_field}" 'valid-pod'
  kube::test::get_object_jsonpath_assert 'pods/valid-pod' "{$id_field}" 'valid-pod'
  # Describe command should print detailed information
  kube::test::describe_object_assert pods 'valid-pod' "Name:" "Image:" "Node:" "Labels:" "Status:"
  # Describe command should print events information by default
  kube::test::describe_object_events_assert pods 'valid-pod'
  # Describe command should not print events information when show-events=false
  kube::test::describe_object_events_assert pods 'valid-pod' false
  # Describe command should print events information when show-events=true
  kube::test::describe_object_events_assert pods 'valid-pod' true
  # Describe command (resource only) should print detailed information
  kube::test::describe_resource_assert pods "Name:" "Image:" "Node:" "Labels:" "Status:"

  # Describe command should print events information by default
  kube::test::describe_resource_events_assert pods
  # Describe command should not print events information when show-events=false
  kube::test::describe_resource_events_assert pods false
  # Describe command should print events information when show-events=true
  kube::test::describe_resource_events_assert pods true
  ### Validate Export ###
  kube::test::get_object_assert 'pods/valid-pod' "{{.metadata.namespace}} {{.metadata.name}}" '<no value> valid-pod' "--export=true"

  ### Dump current valid-pod POD
  output_pod=$(kubectl get pod valid-pod -o yaml --output-version=v1 "${kube_flags[@]}")

  ### Delete POD valid-pod by id
  # Pre-condition: valid-pod POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'
  # Command
  kubectl delete pod valid-pod "${kube_flags[@]}" --grace-period=0 --force
  # Post-condition: valid-pod POD doesn't exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''

  ### Delete POD valid-pod by id with --now
  # Pre-condition: valid-pod POD exists
  kubectl create "${kube_flags[@]}" -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'
  # Command
  kubectl delete pod valid-pod "${kube_flags[@]}" --now
  # Post-condition: valid-pod POD doesn't exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''

  ### Delete POD valid-pod by id with --grace-period=0
  # Pre-condition: valid-pod POD exists
  kubectl create "${kube_flags[@]}" -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'
  # Command succeeds without --force by waiting
  kubectl delete pod valid-pod "${kube_flags[@]}" --grace-period=0
  # Post-condition: valid-pod POD doesn't exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''

  ### Create POD valid-pod from dumped YAML
  # Pre-condition: no POD exists
  create_and_use_new_namespace
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  echo "${output_pod}" | $SED '/namespace:/d' | kubectl create -f - "${kube_flags[@]}"
  # Post-condition: valid-pod POD is created
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'

  ### Delete POD valid-pod from JSON
  # Pre-condition: valid-pod POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'
  # Command
  kubectl delete -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml "${kube_flags[@]}" --grace-period=0 --force
  # Post-condition: valid-pod POD doesn't exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''

  ### Create POD valid-pod from JSON
  # Pre-condition: no POD exists
  create_and_use_new_namespace
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml "${kube_flags[@]}"
  # Post-condition: valid-pod POD is created
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'

  ### Delete POD valid-pod with label
  # Pre-condition: valid-pod POD exists
  kube::test::get_object_assert "pods -l'name in (valid-pod)'" '{{range.items}}{{$id_field}}:{{end}}' 'valid-pod:'
  # Command
  kubectl delete pods -l'name in (valid-pod)' "${kube_flags[@]}" --grace-period=0 --force
  # Post-condition: valid-pod POD doesn't exist
  kube::test::get_object_assert "pods -l'name in (valid-pod)'" '{{range.items}}{{$id_field}}:{{end}}' ''

  ### Create POD valid-pod from YAML
  # Pre-condition: no POD exists
  create_and_use_new_namespace
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml "${kube_flags[@]}"
  # Post-condition: valid-pod POD is created
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'

  ### Delete PODs with no parameter mustn't kill everything
  # Pre-condition: valid-pod POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'
  # Command
  ! kubectl delete pods "${kube_flags[@]}"
  # Post-condition: valid-pod POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'

  ### Delete PODs with --all and a label selector is not permitted
  # Pre-condition: valid-pod POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'
  # Command
  ! kubectl delete --all pods -l'name in (valid-pod)' "${kube_flags[@]}"
  # Post-condition: valid-pod POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'

  ### Delete all PODs
  # Pre-condition: valid-pod POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'
  # Command
  kubectl delete --all pods "${kube_flags[@]}" --grace-period=0 --force # --all remove all the pods
  # Post-condition: no POD exists
  kube::test::get_object_assert "pods -l'name in (valid-pod)'" '{{range.items}}{{$id_field}}:{{end}}' ''

  # Detailed tests for describe pod output
    ### Create a new namespace
  # Pre-condition: the test-secrets namespace does not exist
  kube::test::get_object_assert 'namespaces' '{{range.items}}{{ if eq $id_field \"test-kubectl-describe-pod\" }}found{{end}}{{end}}:' ':'
  # Command
  kubectl create namespace test-kubectl-describe-pod
  # Post-condition: namespace 'test-secrets' is created.
  kube::test::get_object_assert 'namespaces/test-kubectl-describe-pod' "{{$id_field}}" 'test-kubectl-describe-pod'

  ### Create a generic secret
  # Pre-condition: no SECRET exists
  kube::test::get_object_assert 'secrets --namespace=test-kubectl-describe-pod' "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create secret generic test-secret --from-literal=key-1=value1 --type=test-type --namespace=test-kubectl-describe-pod
  # Post-condition: secret exists and has expected values
  kube::test::get_object_assert 'secret/test-secret --namespace=test-kubectl-describe-pod' "{{$id_field}}" 'test-secret'
  kube::test::get_object_assert 'secret/test-secret --namespace=test-kubectl-describe-pod' "{{$secret_type}}" 'test-type'

  ### Create a generic configmap
  # Pre-condition: no CONFIGMAP exists
  kube::test::get_object_assert 'configmaps --namespace=test-kubectl-describe-pod' "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create configmap test-configmap --from-literal=key-2=value2 --namespace=test-kubectl-describe-pod
  # Post-condition: configmap exists and has expected values
  kube::test::get_object_assert 'configmap/test-configmap --namespace=test-kubectl-describe-pod' "{{$id_field}}" 'test-configmap'

  ### Create a pod disruption budget with minAvailable
  # Command
  kubectl create pdb test-pdb-1 --selector=app=rails --min-available=2 --namespace=test-kubectl-describe-pod
  # Post-condition: pdb exists and has expected values
  kube::test::get_object_assert 'pdb/test-pdb-1 --namespace=test-kubectl-describe-pod' "{{$pdb_min_available}}" '2'
  # Command
  kubectl create pdb test-pdb-2 --selector=app=rails --min-available=50% --namespace=test-kubectl-describe-pod
  # Post-condition: pdb exists and has expected values
  kube::test::get_object_assert 'pdb/test-pdb-2 --namespace=test-kubectl-describe-pod' "{{$pdb_min_available}}" '50%'

  ### Create a pod disruption budget with maxUnavailable
  # Command
  kubectl create pdb test-pdb-3 --selector=app=rails --max-unavailable=2 --namespace=test-kubectl-describe-pod
  # Post-condition: pdb exists and has expected values
  kube::test::get_object_assert 'pdb/test-pdb-3 --namespace=test-kubectl-describe-pod' "{{$pdb_max_unavailable}}" '2'
  # Command
  kubectl create pdb test-pdb-4 --selector=app=rails --max-unavailable=50% --namespace=test-kubectl-describe-pod
  # Post-condition: pdb exists and has expected values
  kube::test::get_object_assert 'pdb/test-pdb-4 --namespace=test-kubectl-describe-pod' "{{$pdb_max_unavailable}}" '50%'

  ### Fail creating a pod disruption budget if both maxUnavailable and minAvailable specified
  ! kubectl create pdb test-pdb --selector=app=rails --min-available=2 --max-unavailable=3 --namespace=test-kubectl-describe-pod

  # Create a pod that consumes secret, configmap, and downward API keys as envs
  kube::test::get_object_assert 'pods --namespace=test-kubectl-describe-pod' "{{range.items}}{{$id_field}}:{{end}}" ''
  kubectl create -f hack/testdata/pod-with-api-env.yaml --namespace=test-kubectl-describe-pod

  kube::test::describe_object_assert 'pods --namespace=test-kubectl-describe-pod' 'env-test-pod' "TEST_CMD_1" "<set to the key 'key-1' in secret 'test-secret'>" "TEST_CMD_2" "<set to the key 'key-2' of config map 'test-configmap'>" "TEST_CMD_3" "env-test-pod (v1:metadata.name)"
  # Describe command (resource only) should print detailed information about environment variables
  kube::test::describe_resource_assert 'pods --namespace=test-kubectl-describe-pod' "TEST_CMD_1" "<set to the key 'key-1' in secret 'test-secret'>" "TEST_CMD_2" "<set to the key 'key-2' of config map 'test-configmap'>" "TEST_CMD_3" "env-test-pod (v1:metadata.name)"

  # Clean-up
  kubectl delete pod env-test-pod --namespace=test-kubectl-describe-pod
  kubectl delete secret test-secret --namespace=test-kubectl-describe-pod
  kubectl delete configmap test-configmap --namespace=test-kubectl-describe-pod
  kubectl delete pdb/test-pdb-1 pdb/test-pdb-2 pdb/test-pdb-3 pdb/test-pdb-4 --namespace=test-kubectl-describe-pod
  kubectl delete namespace test-kubectl-describe-pod

  ### Create two PODs
  # Pre-condition: no POD exists
  create_and_use_new_namespace
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml "${kube_flags[@]}"
  kubectl create -f examples/storage/redis/redis-master.yaml "${kube_flags[@]}"
  # Post-condition: valid-pod and redis-master PODs are created
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'redis-master:valid-pod:'

  ### Delete multiple PODs at once
  # Pre-condition: valid-pod and redis-master PODs exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'redis-master:valid-pod:'
  # Command
  kubectl delete pods valid-pod redis-master "${kube_flags[@]}" --grace-period=0 --force # delete multiple pods at once
  # Post-condition: no POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''

  ### Create valid-pod POD
  # Pre-condition: no POD exists
  create_and_use_new_namespace
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml "${kube_flags[@]}"
  # Post-condition: valid-pod POD is created
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'

  ### Label the valid-pod POD
  # Pre-condition: valid-pod is not labelled
  kube::test::get_object_assert 'pod valid-pod' "{{range$labels_field}}{{.}}:{{end}}" 'valid-pod:'
  # Command
  kubectl label pods valid-pod new-name=new-valid-pod "${kube_flags[@]}"
  # Post-condition: valid-pod is labelled
  kube::test::get_object_assert 'pod valid-pod' "{{range$labels_field}}{{.}}:{{end}}" 'valid-pod:new-valid-pod:'

  ### Label the valid-pod POD with empty label value
  # Pre-condition: valid-pod does not have label "emptylabel"
  kube::test::get_object_assert 'pod valid-pod' "{{range$labels_field}}{{.}}:{{end}}" 'valid-pod:new-valid-pod:'
  # Command
  kubectl label pods valid-pod emptylabel="" "${kube_flags[@]}"
  # Post-condition: valid pod contains "emptylabel" with no value
  kube::test::get_object_assert 'pod valid-pod' "{{${labels_field}.emptylabel}}" ''

  ### Annotate the valid-pod POD with empty annotation value
  # Pre-condition: valid-pod does not have annotation "emptyannotation"
  kube::test::get_object_assert 'pod valid-pod' "{{${annotations_field}.emptyannotation}}" '<no value>'
  # Command
  kubectl annotate pods valid-pod emptyannotation="" "${kube_flags[@]}"
  # Post-condition: valid pod contains "emptyannotation" with no value
  kube::test::get_object_assert 'pod valid-pod' "{{${annotations_field}.emptyannotation}}" ''

  ### Record label change
  # Pre-condition: valid-pod does not have record annotation
  kube::test::get_object_assert 'pod valid-pod' "{{range.items}}{{$annotations_field}}:{{end}}" ''
  # Command
  kubectl label pods valid-pod record-change=true --record=true "${kube_flags[@]}"
  # Post-condition: valid-pod has record annotation
  kube::test::get_object_assert 'pod valid-pod' "{{range$annotations_field}}{{.}}:{{end}}" ".*--record=true.*"

  ### Do not record label change
  # Command
  kubectl label pods valid-pod no-record-change=true --record=false "${kube_flags[@]}"
  # Post-condition: valid-pod's record annotation still contains command with --record=true
  kube::test::get_object_assert 'pod valid-pod' "{{range$annotations_field}}{{.}}:{{end}}" ".*--record=true.*"

  ### Record label change with unspecified flag and previous change already recorded
  # Command
  kubectl label pods valid-pod new-record-change=true "${kube_flags[@]}"
  # Post-condition: valid-pod's record annotation contains new change
  kube::test::get_object_assert 'pod valid-pod' "{{range$annotations_field}}{{.}}:{{end}}" ".*new-record-change=true.*"


  ### Delete POD by label
  # Pre-condition: valid-pod POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'
  # Command
  kubectl delete pods -lnew-name=new-valid-pod --grace-period=0 --force "${kube_flags[@]}"
  # Post-condition: valid-pod POD doesn't exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''

  ### Create pod-with-precision POD
  # Pre-condition: no POD is running
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f hack/testdata/pod-with-precision.json "${kube_flags[@]}"
  # Post-condition: valid-pod POD is running
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'pod-with-precision:'

  ## Patch preserves precision
  # Command
  kubectl patch "${kube_flags[@]}" pod pod-with-precision -p='{"metadata":{"annotations":{"patchkey": "patchvalue"}}}'
  # Post-condition: pod-with-precision POD has patched annotation
  kube::test::get_object_assert 'pod pod-with-precision' "{{${annotations_field}.patchkey}}" 'patchvalue'
  # Command
  kubectl label pods pod-with-precision labelkey=labelvalue "${kube_flags[@]}"
  # Post-condition: pod-with-precision POD has label
  kube::test::get_object_assert 'pod pod-with-precision' "{{${labels_field}.labelkey}}" 'labelvalue'
  # Command
  kubectl annotate pods pod-with-precision annotatekey=annotatevalue "${kube_flags[@]}"
  # Post-condition: pod-with-precision POD has annotation
  kube::test::get_object_assert 'pod pod-with-precision' "{{${annotations_field}.annotatekey}}" 'annotatevalue'
  # Cleanup
  kubectl delete pod pod-with-precision "${kube_flags[@]}"

  ### Annotate POD YAML file locally without effecting the live pod.
  kubectl create -f hack/testdata/pod.yaml "${kube_flags[@]}"
  # Command
  kubectl annotate -f hack/testdata/pod.yaml annotatekey=annotatevalue "${kube_flags[@]}"

  # Pre-condition: annotationkey is annotationvalue
  kube::test::get_object_assert 'pod test-pod' "{{${annotations_field}.annotatekey}}" 'annotatevalue'

  # Command
  output_message=$(kubectl annotate --local -f hack/testdata/pod.yaml annotatekey=localvalue -o yaml "${kube_flags[@]}")
  echo $output_message

  # Post-condition: annotationkey is still annotationvalue in the live pod, but command output is the new value
  kube::test::get_object_assert 'pod test-pod' "{{${annotations_field}.annotatekey}}" 'annotatevalue'
  kube::test::if_has_string "${output_message}" "localvalue"

  # Cleanup
  kubectl delete -f hack/testdata/pod.yaml "${kube_flags[@]}"

  ### Create valid-pod POD
  # Pre-condition: no services and no rcs exist
  kube::test::get_object_assert service "{{range.items}}{{$id_field}}:{{end}}" ''
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" ''
  ## kubectl create --edit can update the label filed of multiple resources. tmp-editor.sh is a fake editor
  TEMP=$(mktemp /tmp/tmp-editor-XXXXXXXX.sh)
  echo -e "#!/bin/bash\n$SED -i \"s/mock/modified/g\" \$1" > ${TEMP}
  chmod +x ${TEMP}
  # Command
  EDITOR=${TEMP} kubectl create --edit -f hack/testdata/multi-resource-json.json "${kube_flags[@]}"
  # Post-condition: service named modified and rc named modified are created
  kube::test::get_object_assert service "{{range.items}}{{$id_field}}:{{end}}" 'modified:'
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" 'modified:'
  # Clean up
  kubectl delete service/modified "${kube_flags[@]}"
  kubectl delete rc/modified "${kube_flags[@]}"

  # Pre-condition: no services and no rcs exist
  kube::test::get_object_assert service "{{range.items}}{{$id_field}}:{{end}}" ''
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  EDITOR=${TEMP} kubectl create --edit -f hack/testdata/multi-resource-list.json "${kube_flags[@]}"
  # Post-condition: service named modified and rc named modified are created
  kube::test::get_object_assert service "{{range.items}}{{$id_field}}:{{end}}" 'modified:'
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" 'modified:'
  # Clean up
  rm ${TEMP}
  kubectl delete service/modified "${kube_flags[@]}"
  kubectl delete rc/modified "${kube_flags[@]}"

  ## kubectl create --edit won't create anything if user makes no changes
  [ "$(EDITOR=cat kubectl create --edit -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml -o json 2>&1 | grep 'Edit cancelled')" ]

  ## Create valid-pod POD
  # Pre-condition: no POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml "${kube_flags[@]}"
  # Post-condition: valid-pod POD is created
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'

  ## Patch can modify a local object
  kubectl patch --local -f pkg/kubectl/validation/testdata/v1/validPod.yaml --patch='{"spec": {"restartPolicy":"Never"}}' -o jsonpath='{.spec.restartPolicy}' | grep -q "Never"

  ## Patch pod can change image
  # Command
  kubectl patch "${kube_flags[@]}" pod valid-pod --record -p='{"spec":{"containers":[{"name": "kubernetes-serve-hostname", "image": "nginx"}]}}'
  # Post-condition: valid-pod POD has image nginx
  kube::test::get_object_assert pods "{{range.items}}{{$image_field}}:{{end}}" 'nginx:'
  # Post-condition: valid-pod has the record annotation
  kube::test::get_object_assert pods "{{range.items}}{{$annotations_field}}:{{end}}" "${change_cause_annotation}"
  # prove that patch can use different types
  kubectl patch "${kube_flags[@]}" pod valid-pod --type="json" -p='[{"op": "replace", "path": "/spec/containers/0/image", "value":"nginx2"}]'
  # Post-condition: valid-pod POD has image nginx
  kube::test::get_object_assert pods "{{range.items}}{{$image_field}}:{{end}}" 'nginx2:'
  # prove that patch can use different types
  kubectl patch "${kube_flags[@]}" pod valid-pod --type="json" -p='[{"op": "replace", "path": "/spec/containers/0/image", "value":"nginx"}]'
  # Post-condition: valid-pod POD has image nginx
  kube::test::get_object_assert pods "{{range.items}}{{$image_field}}:{{end}}" 'nginx:'
  # prove that yaml input works too
  YAML_PATCH=$'spec:\n  containers:\n  - name: kubernetes-serve-hostname\n    image: changed-with-yaml\n'
  kubectl patch "${kube_flags[@]}" pod valid-pod -p="${YAML_PATCH}"
  # Post-condition: valid-pod POD has image nginx
  kube::test::get_object_assert pods "{{range.items}}{{$image_field}}:{{end}}" 'changed-with-yaml:'
  ## Patch pod from JSON can change image
  # Command
  kubectl patch "${kube_flags[@]}" -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml -p='{"spec":{"containers":[{"name": "kubernetes-serve-hostname", "image": "gcr.io/google_containers/pause-amd64:3.0"}]}}'
  # Post-condition: valid-pod POD has image gcr.io/google_containers/pause-amd64:3.0
  kube::test::get_object_assert pods "{{range.items}}{{$image_field}}:{{end}}" 'gcr.io/google_containers/pause-amd64:3.0:'

  ## If resourceVersion is specified in the patch, it will be treated as a precondition, i.e., if the resourceVersion is different from that is stored in the server, the Patch should be rejected
  ERROR_FILE="${KUBE_TEMP}/conflict-error"
  ## If the resourceVersion is the same as the one stored in the server, the patch will be applied.
  # Command
  # Needs to retry because other party may change the resource.
  for count in {0..3}; do
    resourceVersion=$(kubectl get "${kube_flags[@]}" pod valid-pod -o go-template='{{ .metadata.resourceVersion }}')
    kubectl patch "${kube_flags[@]}" pod valid-pod -p='{"spec":{"containers":[{"name": "kubernetes-serve-hostname", "image": "nginx"}]},"metadata":{"resourceVersion":"'$resourceVersion'"}}' 2> "${ERROR_FILE}" || true
    if grep -q "the object has been modified" "${ERROR_FILE}"; then
      kube::log::status "retry $1, error: $(cat ${ERROR_FILE})"
      rm "${ERROR_FILE}"
      sleep $((2**count))
    else
      rm "${ERROR_FILE}"
      kube::test::get_object_assert pods "{{range.items}}{{$image_field}}:{{end}}" 'nginx:'
      break
    fi
  done

  ## If the resourceVersion is the different from the one stored in the server, the patch will be rejected.
  resourceVersion=$(kubectl get "${kube_flags[@]}" pod valid-pod -o go-template='{{ .metadata.resourceVersion }}')
  ((resourceVersion+=100))
  # Command
  kubectl patch "${kube_flags[@]}" pod valid-pod -p='{"spec":{"containers":[{"name": "kubernetes-serve-hostname", "image": "nginx"}]},"metadata":{"resourceVersion":"'$resourceVersion'"}}' 2> "${ERROR_FILE}" || true
  # Post-condition: should get an error reporting the conflict
  if grep -q "please apply your changes to the latest version and try again" "${ERROR_FILE}"; then
    kube::log::status "\"kubectl patch with resourceVersion $resourceVersion\" returns error as expected: $(cat ${ERROR_FILE})"
  else
    kube::log::status "\"kubectl patch with resourceVersion $resourceVersion\" returns unexpected error or non-error: $(cat ${ERROR_FILE})"
    exit 1
  fi
  rm "${ERROR_FILE}"

  ## --force replace pod can change other field, e.g., spec.container.name
  # Command
  kubectl get "${kube_flags[@]}" pod valid-pod -o json | $SED 's/"kubernetes-serve-hostname"/"replaced-k8s-serve-hostname"/g' > /tmp/tmp-valid-pod.json
  kubectl replace "${kube_flags[@]}" --force -f /tmp/tmp-valid-pod.json
  # Post-condition: spec.container.name = "replaced-k8s-serve-hostname"
  kube::test::get_object_assert 'pod valid-pod' "{{(index .spec.containers 0).name}}" 'replaced-k8s-serve-hostname'

  ## check replace --grace-period requires --force
  output_message=$(! kubectl replace "${kube_flags[@]}" --grace-period=1 -f /tmp/tmp-valid-pod.json 2>&1)
  kube::test::if_has_string "${output_message}" '\-\-grace-period must have \-\-force specified'

  ## check replace --timeout requires --force
  output_message=$(! kubectl replace "${kube_flags[@]}" --timeout=1s -f /tmp/tmp-valid-pod.json 2>&1)
  kube::test::if_has_string "${output_message}" '\-\-timeout must have \-\-force specified'

  #cleaning
  rm /tmp/tmp-valid-pod.json

  ## replace of a cluster scoped resource can succeed
  # Pre-condition: a node exists
  kubectl create -f - "${kube_flags[@]}" << __EOF__
{
  "kind": "Node",
  "apiVersion": "v1",
  "metadata": {
    "name": "node-v1-test"
  }
}
__EOF__
  kubectl replace -f - "${kube_flags[@]}" << __EOF__
{
  "kind": "Node",
  "apiVersion": "v1",
  "metadata": {
    "name": "node-v1-test",
    "annotations": {"a":"b"},
    "resourceVersion": "0"
  }
}
__EOF__

  # Post-condition: the node command succeeds
  kube::test::get_object_assert "node node-v1-test" "{{.metadata.annotations.a}}" 'b'
  kubectl delete node node-v1-test "${kube_flags[@]}"

  ## kubectl edit can update the image field of a POD. tmp-editor.sh is a fake editor
  echo -e "#!/bin/bash\n$SED -i \"s/nginx/gcr.io\/google_containers\/serve_hostname/g\" \$1" > /tmp/tmp-editor.sh
  chmod +x /tmp/tmp-editor.sh
  # Pre-condition: valid-pod POD has image nginx
  kube::test::get_object_assert pods "{{range.items}}{{$image_field}}:{{end}}" 'nginx:'
  [[ "$(EDITOR=/tmp/tmp-editor.sh kubectl edit "${kube_flags[@]}" pods/valid-pod --output-patch=true | grep Patch:)" ]]
  # Post-condition: valid-pod POD has image gcr.io/google_containers/serve_hostname
  kube::test::get_object_assert pods "{{range.items}}{{$image_field}}:{{end}}" 'gcr.io/google_containers/serve_hostname:'
  # cleaning
  rm /tmp/tmp-editor.sh

  ## kubectl edit should work on Windows
  [ "$(EDITOR=cat kubectl edit pod/valid-pod 2>&1 | grep 'Edit cancelled')" ]
  [ "$(EDITOR=cat kubectl edit pod/valid-pod | grep 'name: valid-pod')" ]
  [ "$(EDITOR=cat kubectl edit --windows-line-endings pod/valid-pod | file - | grep CRLF)" ]
  [ ! "$(EDITOR=cat kubectl edit --windows-line-endings=false pod/valid-pod | file - | grep CRLF)" ]
  [ "$(EDITOR=cat kubectl edit ns | grep 'kind: List')" ]

  ### Label POD YAML file locally without effecting the live pod.
  # Pre-condition: name is valid-pod
  kube::test::get_object_assert 'pod valid-pod' "{{${labels_field}.name}}" 'valid-pod'
  # Command
  output_message=$(kubectl label --local --overwrite -f hack/testdata/pod.yaml name=localonlyvalue -o yaml "${kube_flags[@]}")
  echo $output_message
  # Post-condition: name is still valid-pod in the live pod, but command output is the new value
  kube::test::get_object_assert 'pod valid-pod' "{{${labels_field}.name}}" 'valid-pod'
  kube::test::if_has_string "${output_message}" "localonlyvalue"

  ### Overwriting an existing label is not permitted
  # Pre-condition: name is valid-pod
  kube::test::get_object_assert 'pod valid-pod' "{{${labels_field}.name}}" 'valid-pod'
  # Command
  ! kubectl label pods valid-pod name=valid-pod-super-sayan "${kube_flags[@]}"
  # Post-condition: name is still valid-pod
  kube::test::get_object_assert 'pod valid-pod' "{{${labels_field}.name}}" 'valid-pod'

  ### --overwrite must be used to overwrite existing label, can be applied to all resources
  # Pre-condition: name is valid-pod
  kube::test::get_object_assert 'pod valid-pod' "{{${labels_field}.name}}" 'valid-pod'
  # Command
  kubectl label --overwrite pods --all name=valid-pod-super-sayan "${kube_flags[@]}"
  # Post-condition: name is valid-pod-super-sayan
  kube::test::get_object_assert 'pod valid-pod' "{{${labels_field}.name}}" 'valid-pod-super-sayan'

  ### Delete POD by label
  # Pre-condition: valid-pod POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'
  # Command
  kubectl delete pods -l'name in (valid-pod-super-sayan)' --grace-period=0 --force "${kube_flags[@]}"
  # Post-condition: valid-pod POD doesn't exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''

  ### Create two PODs from 1 yaml file
  # Pre-condition: no POD exists
  create_and_use_new_namespace
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f test/fixtures/doc-yaml/user-guide/multi-pod.yaml "${kube_flags[@]}"
  # Post-condition: redis-master and valid-pod PODs exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'redis-master:valid-pod:'

  ### Delete two PODs from 1 yaml file
  # Pre-condition: redis-master and valid-pod PODs exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'redis-master:valid-pod:'
  # Command
  kubectl delete -f test/fixtures/doc-yaml/user-guide/multi-pod.yaml "${kube_flags[@]}"
  # Post-condition: no PODs exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''

  ## kubectl apply should update configuration annotations only if apply is already called
  ## 1. kubectl create doesn't set the annotation
  # Pre-Condition: no POD exists
  create_and_use_new_namespace
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command: create a pod "test-pod"
  kubectl create -f hack/testdata/pod.yaml "${kube_flags[@]}"
  # Post-Condition: pod "test-pod" is created
  kube::test::get_object_assert 'pods test-pod' "{{${labels_field}.name}}" 'test-pod-label'
  # Post-Condition: pod "test-pod" doesn't have configuration annotation
  ! [[ "$(kubectl get pods test-pod -o yaml "${kube_flags[@]}" | grep kubectl.kubernetes.io/last-applied-configuration)" ]]
  ## 2. kubectl replace doesn't set the annotation
  kubectl get pods test-pod -o yaml "${kube_flags[@]}" | $SED 's/test-pod-label/test-pod-replaced/g' > "${KUBE_TEMP}"/test-pod-replace.yaml
  # Command: replace the pod "test-pod"
  kubectl replace -f "${KUBE_TEMP}"/test-pod-replace.yaml "${kube_flags[@]}"
  # Post-Condition: pod "test-pod" is replaced
  kube::test::get_object_assert 'pods test-pod' "{{${labels_field}.name}}" 'test-pod-replaced'
  # Post-Condition: pod "test-pod" doesn't have configuration annotation
  ! [[ "$(kubectl get pods test-pod -o yaml "${kube_flags[@]}" | grep kubectl.kubernetes.io/last-applied-configuration)" ]]
  ## 3. kubectl apply does set the annotation
  # Command: apply the pod "test-pod"
  kubectl apply -f hack/testdata/pod-apply.yaml "${kube_flags[@]}"
  # Post-Condition: pod "test-pod" is applied
  kube::test::get_object_assert 'pods test-pod' "{{${labels_field}.name}}" 'test-pod-applied'
  # Post-Condition: pod "test-pod" has configuration annotation
  [[ "$(kubectl get pods test-pod -o yaml "${kube_flags[@]}" | grep kubectl.kubernetes.io/last-applied-configuration)" ]]
  kubectl get pods test-pod -o yaml "${kube_flags[@]}" | grep kubectl.kubernetes.io/last-applied-configuration > "${KUBE_TEMP}"/annotation-configuration
  ## 4. kubectl replace updates an existing annotation
  kubectl get pods test-pod -o yaml "${kube_flags[@]}" | $SED 's/test-pod-applied/test-pod-replaced/g' > "${KUBE_TEMP}"/test-pod-replace.yaml
  # Command: replace the pod "test-pod"
  kubectl replace -f "${KUBE_TEMP}"/test-pod-replace.yaml "${kube_flags[@]}"
  # Post-Condition: pod "test-pod" is replaced
  kube::test::get_object_assert 'pods test-pod' "{{${labels_field}.name}}" 'test-pod-replaced'
  # Post-Condition: pod "test-pod" has configuration annotation, and it's updated (different from the annotation when it's applied)
  [[ "$(kubectl get pods test-pod -o yaml "${kube_flags[@]}" | grep kubectl.kubernetes.io/last-applied-configuration)" ]]
  kubectl get pods test-pod -o yaml "${kube_flags[@]}" | grep kubectl.kubernetes.io/last-applied-configuration > "${KUBE_TEMP}"/annotation-configuration-replaced
  ! [[ $(diff -q "${KUBE_TEMP}"/annotation-configuration "${KUBE_TEMP}"/annotation-configuration-replaced > /dev/null) ]]
  # Clean up
  rm "${KUBE_TEMP}"/test-pod-replace.yaml "${KUBE_TEMP}"/annotation-configuration "${KUBE_TEMP}"/annotation-configuration-replaced
  kubectl delete pods test-pod "${kube_flags[@]}"

  set +o nounset
  set +o errexit
}

# Runs tests related to kubectl apply.
run_kubectl_apply_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl apply"
  ## kubectl apply should create the resource that doesn't exist yet
  # Pre-Condition: no POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command: apply a pod "test-pod" (doesn't exist) should create this pod
  kubectl apply -f hack/testdata/pod.yaml "${kube_flags[@]}"
  # Post-Condition: pod "test-pod" is created
  kube::test::get_object_assert 'pods test-pod' "{{${labels_field}.name}}" 'test-pod-label'
  # Post-Condition: pod "test-pod" has configuration annotation
  [[ "$(kubectl get pods test-pod -o yaml "${kube_flags[@]}" | grep kubectl.kubernetes.io/last-applied-configuration)" ]]
  # Clean up
  kubectl delete pods test-pod "${kube_flags[@]}"


  ## kubectl apply should be able to clear defaulted fields.
  # Pre-Condition: no deployment exists
  kube::test::get_object_assert deployments "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command: apply a deployment "test-deployment-retainkeys" (doesn't exist) should create this deployment
  kubectl apply -f hack/testdata/retainKeys/deployment/deployment-before.yaml "${kube_flags[@]}"
  # Post-Condition: deployment "test-deployment-retainkeys" created
  kube::test::get_object_assert deployments "{{range.items}}{{$id_field}}{{end}}" 'test-deployment-retainkeys'
  # Post-Condition: deployment "test-deployment-retainkeys" has defaulted fields
  [[ "$(kubectl get deployments test-deployment-retainkeys -o yaml "${kube_flags[@]}" | grep RollingUpdate)" ]]
  [[ "$(kubectl get deployments test-deployment-retainkeys -o yaml "${kube_flags[@]}" | grep maxSurge)" ]]
  [[ "$(kubectl get deployments test-deployment-retainkeys -o yaml "${kube_flags[@]}" | grep maxUnavailable)" ]]
  [[ "$(kubectl get deployments test-deployment-retainkeys -o yaml "${kube_flags[@]}" | grep emptyDir)" ]]
  # Command: apply a deployment "test-deployment-retainkeys" should clear
  # defaulted fields and successfully update the deployment
  [[ "$(kubectl apply -f hack/testdata/retainKeys/deployment/deployment-after.yaml "${kube_flags[@]}")" ]]
  # Post-Condition: deployment "test-deployment-retainkeys" has updated fields
  [[ "$(kubectl get deployments test-deployment-retainkeys -o yaml "${kube_flags[@]}" | grep Recreate)" ]]
  ! [[ "$(kubectl get deployments test-deployment-retainkeys -o yaml "${kube_flags[@]}" | grep RollingUpdate)" ]]
  [[ "$(kubectl get deployments test-deployment-retainkeys -o yaml "${kube_flags[@]}" | grep hostPath)" ]]
  ! [[ "$(kubectl get deployments test-deployment-retainkeys -o yaml "${kube_flags[@]}" | grep emptyDir)" ]]
  # Clean up
  kubectl delete deployments test-deployment-retainkeys "${kube_flags[@]}"


  ## kubectl apply -f with label selector should only apply matching objects
  # Pre-Condition: no POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # apply
  kubectl apply -l unique-label=bingbang -f hack/testdata/filter "${kube_flags[@]}"
  # check right pod exists
  kube::test::get_object_assert 'pods selector-test-pod' "{{${labels_field}.name}}" 'selector-test-pod'
  # check wrong pod doesn't exist
  output_message=$(! kubectl get pods selector-test-pod-dont-apply 2>&1 "${kube_flags[@]}")
  kube::test::if_has_string "${output_message}" 'pods "selector-test-pod-dont-apply" not found'
  # cleanup
  kubectl delete pods selector-test-pod


  ## kubectl apply --prune
  # Pre-Condition: no POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''

  # apply a
  kubectl apply --prune -l prune-group=true -f hack/testdata/prune/a.yaml "${kube_flags[@]}"
  # check right pod exists
  kube::test::get_object_assert 'pods a' "{{${id_field}}}" 'a'
  # check wrong pod doesn't exist
  output_message=$(! kubectl get pods b 2>&1 "${kube_flags[@]}")
  kube::test::if_has_string "${output_message}" 'pods "b" not found'

  # apply b
  kubectl apply --prune -l prune-group=true -f hack/testdata/prune/b.yaml "${kube_flags[@]}"
  # check right pod exists
  kube::test::get_object_assert 'pods b' "{{${id_field}}}" 'b'
  # check wrong pod doesn't exist
  output_message=$(! kubectl get pods a 2>&1 "${kube_flags[@]}")
  kube::test::if_has_string "${output_message}" 'pods "a" not found'

  # cleanup
  kubectl delete pods b

  # same thing without prune for a sanity check
  # Pre-Condition: no POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''

  # apply a
  kubectl apply -l prune-group=true -f hack/testdata/prune/a.yaml "${kube_flags[@]}"
  # check right pod exists
  kube::test::get_object_assert 'pods a' "{{${id_field}}}" 'a'
  # check wrong pod doesn't exist
  output_message=$(! kubectl get pods b 2>&1 "${kube_flags[@]}")
  kube::test::if_has_string "${output_message}" 'pods "b" not found'

  # apply b
  kubectl apply -l prune-group=true -f hack/testdata/prune/b.yaml "${kube_flags[@]}"
  # check both pods exist
  kube::test::get_object_assert 'pods a' "{{${id_field}}}" 'a'
  kube::test::get_object_assert 'pods b' "{{${id_field}}}" 'b'
  # check wrong pod doesn't exist

  # cleanup
  kubectl delete pod/a pod/b

  ## kubectl apply --prune requires a --all flag to select everything
  output_message=$(! kubectl apply --prune -f hack/testdata/prune 2>&1 "${kube_flags[@]}")
  kube::test::if_has_string "${output_message}" \
    'all resources selected for prune without explicitly passing --all'
  # should apply everything
  kubectl apply --all --prune -f hack/testdata/prune
  kube::test::get_object_assert 'pods a' "{{${id_field}}}" 'a'
  kube::test::get_object_assert 'pods b' "{{${id_field}}}" 'b'
  kubectl delete pod/a pod/b

  ## kubectl apply --prune should fallback to delete for non reapable types
  kubectl apply --all --prune -f hack/testdata/prune-reap/a.yml 2>&1 "${kube_flags[@]}"
  kube::test::get_object_assert 'pvc a-pvc' "{{${id_field}}}" 'a-pvc'
  kubectl apply --all --prune -f hack/testdata/prune-reap/b.yml 2>&1 "${kube_flags[@]}"
  kube::test::get_object_assert 'pvc b-pvc' "{{${id_field}}}" 'b-pvc'
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  kubectl delete pvc b-pvc 2>&1 "${kube_flags[@]}"

  ## kubectl apply --prune --prune-whitelist
  # Pre-Condition: no POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # apply pod a
  kubectl apply --prune -l prune-group=true -f hack/testdata/prune/a.yaml "${kube_flags[@]}"
  # check right pod exists
  kube::test::get_object_assert 'pods a' "{{${id_field}}}" 'a'
  # apply svc and don't prune pod a by overwriting whitelist
  kubectl apply --prune -l prune-group=true -f hack/testdata/prune/svc.yaml --prune-whitelist core/v1/Service 2>&1 "${kube_flags[@]}"
  kube::test::get_object_assert 'service prune-svc' "{{${id_field}}}" 'prune-svc'
  kube::test::get_object_assert 'pods a' "{{${id_field}}}" 'a'
  # apply svc and prune pod a with default whitelist
  kubectl apply --prune -l prune-group=true -f hack/testdata/prune/svc.yaml 2>&1 "${kube_flags[@]}"
  kube::test::get_object_assert 'service prune-svc' "{{${id_field}}}" 'prune-svc'
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # cleanup
  kubectl delete svc prune-svc 2>&1 "${kube_flags[@]}"

  set +o nounset
  set +o errexit
}

# Runs tests related to kubectl create --filename(-f) --selector(-l).
run_kubectl_create_filter_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl create filter"
  ## kubectl create -f with label selector should only create matching objects
  # Pre-Condition: no POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # create
  kubectl create -l unique-label=bingbang -f hack/testdata/filter "${kube_flags[@]}"
  # check right pod exists
  kube::test::get_object_assert 'pods selector-test-pod' "{{${labels_field}.name}}" 'selector-test-pod'
  # check wrong pod doesn't exist
  output_message=$(! kubectl get pods selector-test-pod-dont-apply 2>&1 "${kube_flags[@]}")
  kube::test::if_has_string "${output_message}" 'pods "selector-test-pod-dont-apply" not found'
  # cleanup
  kubectl delete pods selector-test-pod

  set +o nounset
  set +o errexit
}

run_kubectl_apply_deployments_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl apply deployments"
  ## kubectl apply should propagate user defined null values
  # Pre-Condition: no Deployments, ReplicaSets, Pods exist
  kube::test::get_object_assert deployments "{{range.items}}{{$id_field}}:{{end}}" ''
  kube::test::get_object_assert replicasets "{{range.items}}{{$id_field}}:{{end}}" ''
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # apply base deployment
  kubectl apply -f hack/testdata/null-propagation/deployment-l1.yaml "${kube_flags[@]}"
  # check right deployment exists
  kube::test::get_object_assert 'deployments my-depl' "{{${id_field}}}" 'my-depl'
  # check right labels exists
  kube::test::get_object_assert 'deployments my-depl' "{{.spec.template.metadata.labels.l1}}" 'l1'
  kube::test::get_object_assert 'deployments my-depl' "{{.spec.selector.matchLabels.l1}}" 'l1'
  kube::test::get_object_assert 'deployments my-depl' "{{.metadata.labels.l1}}" 'l1'

  # apply new deployment with new template labels
  kubectl apply -f hack/testdata/null-propagation/deployment-l2.yaml "${kube_flags[@]}"
  # check right labels exists
  kube::test::get_object_assert 'deployments my-depl' "{{.spec.template.metadata.labels.l1}}" '<no value>'
  kube::test::get_object_assert 'deployments my-depl' "{{.spec.selector.matchLabels.l1}}" '<no value>'
  kube::test::get_object_assert 'deployments my-depl' "{{.metadata.labels.l1}}" '<no value>'
  kube::test::get_object_assert 'deployments my-depl' "{{.spec.template.metadata.labels.l2}}" 'l2'
  kube::test::get_object_assert 'deployments my-depl' "{{.spec.selector.matchLabels.l2}}" 'l2'
  kube::test::get_object_assert 'deployments my-depl' "{{.metadata.labels.l2}}" 'l2'

  # cleanup
  # need to explicitly remove replicasets and pods because we changed the deployment selector and orphaned things
  kubectl delete deployments,rs,pods --all --cascade=false --grace-period=0
  # Post-Condition: no Deployments, ReplicaSets, Pods exist
  kube::test::wait_object_assert deployments "{{range.items}}{{$id_field}}:{{end}}" ''
  kube::test::wait_object_assert replicasets "{{range.items}}{{$id_field}}:{{end}}" ''
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''

  set +o nounset
  set +o errexit
}

# Runs tests for --save-config tests.
run_save_config_tests() {
  set -o nounset
  set -o errexit

  kube::log::status "Testing kubectl --save-config"
  ## Configuration annotations should be set when --save-config is enabled
  ## 1. kubectl create --save-config should generate configuration annotation
  # Pre-Condition: no POD exists
  create_and_use_new_namespace
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command: create a pod "test-pod"
  kubectl create -f hack/testdata/pod.yaml --save-config "${kube_flags[@]}"
  # Post-Condition: pod "test-pod" has configuration annotation
  [[ "$(kubectl get pods test-pod -o yaml "${kube_flags[@]}" | grep kubectl.kubernetes.io/last-applied-configuration)" ]]
  # Clean up
  kubectl delete -f hack/testdata/pod.yaml "${kube_flags[@]}"
  ## 2. kubectl edit --save-config should generate configuration annotation
  # Pre-Condition: no POD exists, then create pod "test-pod", which shouldn't have configuration annotation
  create_and_use_new_namespace
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  kubectl create -f hack/testdata/pod.yaml "${kube_flags[@]}"
  ! [[ "$(kubectl get pods test-pod -o yaml "${kube_flags[@]}" | grep kubectl.kubernetes.io/last-applied-configuration)" ]]
  # Command: edit the pod "test-pod"
  temp_editor="${KUBE_TEMP}/tmp-editor.sh"
  echo -e "#!/bin/bash\n$SED -i \"s/test-pod-label/test-pod-label-edited/g\" \$@" > "${temp_editor}"
  chmod +x "${temp_editor}"
  EDITOR=${temp_editor} kubectl edit pod test-pod --save-config "${kube_flags[@]}"
  # Post-Condition: pod "test-pod" has configuration annotation
  [[ "$(kubectl get pods test-pod -o yaml "${kube_flags[@]}" | grep kubectl.kubernetes.io/last-applied-configuration)" ]]
  # Clean up
  kubectl delete -f hack/testdata/pod.yaml "${kube_flags[@]}"
  ## 3. kubectl replace --save-config should generate configuration annotation
  # Pre-Condition: no POD exists, then create pod "test-pod", which shouldn't have configuration annotation
  create_and_use_new_namespace
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  kubectl create -f hack/testdata/pod.yaml "${kube_flags[@]}"
  ! [[ "$(kubectl get pods test-pod -o yaml "${kube_flags[@]}" | grep kubectl.kubernetes.io/last-applied-configuration)" ]]
  # Command: replace the pod "test-pod"
  kubectl replace -f hack/testdata/pod.yaml --save-config "${kube_flags[@]}"
  # Post-Condition: pod "test-pod" has configuration annotation
  [[ "$(kubectl get pods test-pod -o yaml "${kube_flags[@]}" | grep kubectl.kubernetes.io/last-applied-configuration)" ]]
  # Clean up
  kubectl delete -f hack/testdata/pod.yaml "${kube_flags[@]}"
  ## 4. kubectl run --save-config should generate configuration annotation
  # Pre-Condition: no RC exists
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command: create the rc "nginx" with image nginx
  kubectl run nginx "--image=$IMAGE_NGINX" --save-config --generator=run/v1 "${kube_flags[@]}"
  # Post-Condition: rc "nginx" has configuration annotation
  [[ "$(kubectl get rc nginx -o yaml "${kube_flags[@]}" | grep kubectl.kubernetes.io/last-applied-configuration)" ]]
  ## 5. kubectl expose --save-config should generate configuration annotation
  # Pre-Condition: no service exists
  kube::test::get_object_assert svc "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command: expose the rc "nginx"
  kubectl expose rc nginx --save-config --port=80 --target-port=8000 "${kube_flags[@]}"
  # Post-Condition: service "nginx" has configuration annotation
  [[ "$(kubectl get svc nginx -o yaml "${kube_flags[@]}" | grep kubectl.kubernetes.io/last-applied-configuration)" ]]
  # Clean up
  kubectl delete rc,svc nginx
  ## 6. kubectl autoscale --save-config should generate configuration annotation
  # Pre-Condition: no RC exists, then create the rc "frontend", which shouldn't have configuration annotation
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" ''
  kubectl create -f hack/testdata/frontend-controller.yaml "${kube_flags[@]}"
  ! [[ "$(kubectl get rc frontend -o yaml "${kube_flags[@]}" | grep kubectl.kubernetes.io/last-applied-configuration)" ]]
  # Command: autoscale rc "frontend"
  kubectl autoscale -f hack/testdata/frontend-controller.yaml --save-config "${kube_flags[@]}" --max=2
  # Post-Condition: hpa "frontend" has configuration annotation
  [[ "$(kubectl get hpa frontend -o yaml "${kube_flags[@]}" | grep kubectl.kubernetes.io/last-applied-configuration)" ]]
  # Ensure we can interact with HPA objects in lists through autoscaling/v1 APIs
  output_message=$(kubectl get hpa -o=jsonpath='{.items[0].apiVersion}' 2>&1 "${kube_flags[@]}")
  kube::test::if_has_string "${output_message}" 'autoscaling/v1'
  output_message=$(kubectl get hpa.autoscaling -o=jsonpath='{.items[0].apiVersion}' 2>&1 "${kube_flags[@]}")
  kube::test::if_has_string "${output_message}" 'autoscaling/v1'
  # tests kubectl group prefix matching
  output_message=$(kubectl get hpa.autoscal -o=jsonpath='{.items[0].apiVersion}' 2>&1 "${kube_flags[@]}")
  kube::test::if_has_string "${output_message}" 'autoscaling/v1'
  # Clean up
  # Note that we should delete hpa first, otherwise it may fight with the rc reaper.
  kubectl delete hpa frontend "${kube_flags[@]}"
  kubectl delete rc  frontend "${kube_flags[@]}"

  set +o nounset
  set +o errexit
}

run_kubectl_run_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl run"
  ## kubectl run should create deployments, jobs or cronjob
  # Pre-Condition: no Job exists
  kube::test::get_object_assert jobs "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl run pi --generator=job/v1 "--image=$IMAGE_PERL" --restart=OnFailure -- perl -Mbignum=bpi -wle 'print bpi(20)' "${kube_flags[@]}"
  # Post-Condition: Job "pi" is created
  kube::test::get_object_assert jobs "{{range.items}}{{$id_field}}:{{end}}" 'pi:'
  # Describe command (resource only) should print detailed information
  kube::test::describe_resource_assert pods "Name:" "Image:" "Node:" "Labels:" "Status:" "Created By"
  # Clean up
  kubectl delete jobs pi "${kube_flags[@]}"
  # Post-condition: no pods exist.
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''

  # Pre-Condition: no Deployment exists
  kube::test::get_object_assert deployment "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl run nginx-extensions "--image=$IMAGE_NGINX" "${kube_flags[@]}"
  # Post-Condition: Deployment "nginx" is created
  kube::test::get_object_assert deployment.extensions "{{range.items}}{{$id_field}}:{{end}}" 'nginx-extensions:'
  # and old generator was used, iow. old defaults are applied
  output_message=$(kubectl get deployment.extensions/nginx-extensions -o jsonpath='{.spec.revisionHistoryLimit}')
  kube::test::if_has_not_string "${output_message}" '2'
  # Clean up
  kubectl delete deployment nginx-extensions "${kube_flags[@]}"
  # Command
  kubectl run nginx-apps "--image=$IMAGE_NGINX" --generator=deployment/apps.v1beta1 "${kube_flags[@]}"
  # Post-Condition: Deployment "nginx" is created
  kube::test::get_object_assert deployment.apps "{{range.items}}{{$id_field}}:{{end}}" 'nginx-apps:'
  # and new generator was used, iow. new defaults are applied
  output_message=$(kubectl get deployment/nginx-apps -o jsonpath='{.spec.revisionHistoryLimit}')
  kube::test::if_has_string "${output_message}" '2'
  # Clean up
  kubectl delete deployment nginx-apps "${kube_flags[@]}"

  # Pre-Condition: no Job exists
  kube::test::get_object_assert cronjobs "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl run pi --schedule="*/5 * * * *" --generator=cronjob/v1beta1 "--image=$IMAGE_PERL" --restart=OnFailure -- perl -Mbignum=bpi -wle 'print bpi(20)' "${kube_flags[@]}"
  # Post-Condition: CronJob "pi" is created
  kube::test::get_object_assert cronjobs "{{range.items}}{{$id_field}}:{{end}}" 'pi:'
  # Clean up
  kubectl delete cronjobs pi "${kube_flags[@]}"

  set +o nounset
  set +o errexit
}

run_kubectl_get_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl get"
  ### Test retrieval of non-existing pods
  # Pre-condition: no POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  output_message=$(! kubectl get pods abc 2>&1 "${kube_flags[@]}")
  # Post-condition: POD abc should error since it doesn't exist
  kube::test::if_has_string "${output_message}" 'pods "abc" not found'

  ### Test retrieval of non-existing POD with output flag specified
  # Pre-condition: no POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  output_message=$(! kubectl get pods abc 2>&1 "${kube_flags[@]}" -o name)
  # Post-condition: POD abc should error since it doesn't exist
  kube::test::if_has_string "${output_message}" 'pods "abc" not found'

  ### Test retrieval of pods when none exist with non-human readable output format flag specified
  # Pre-condition: no pods exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  output_message=$(kubectl get pods 2>&1 "${kube_flags[@]}" -o json)
  # Post-condition: The text "No resources found" should not be part of the output
  kube::test::if_has_not_string "${output_message}" 'No resources found'
  # Command
  output_message=$(kubectl get pods 2>&1 "${kube_flags[@]}" -o yaml)
  # Post-condition: The text "No resources found" should not be part of the output
  kube::test::if_has_not_string "${output_message}" 'No resources found'
  # Command
  output_message=$(kubectl get pods 2>&1 "${kube_flags[@]}" -o name)
  # Post-condition: The text "No resources found" should not be part of the output
  kube::test::if_has_not_string "${output_message}" 'No resources found'
  # Command
  output_message=$(kubectl get pods 2>&1 "${kube_flags[@]}" -o jsonpath='{.items}')
  # Post-condition: The text "No resources found" should not be part of the output
  kube::test::if_has_not_string "${output_message}" 'No resources found'
  # Command
  output_message=$(kubectl get pods 2>&1 "${kube_flags[@]}" -o go-template='{{.items}}')
  # Post-condition: The text "No resources found" should not be part of the output
  kube::test::if_has_not_string "${output_message}" 'No resources found'
  # Command
  output_message=$(kubectl get pods 2>&1 "${kube_flags[@]}" -o custom-columns=NAME:.metadata.name)
  # Post-condition: The text "No resources found" should not be part of the output
  kube::test::if_has_not_string "${output_message}" 'No resources found'

  ### Test retrieval of pods when none exist, with human-readable output format flag specified
  # Pre-condition: no pods exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  output_message=$(kubectl get pods 2>&1 "${kube_flags[@]}")
  # Post-condition: The text "No resources found" should be part of the output
  kube::test::if_has_string "${output_message}" 'No resources found'
  # Command
  output_message=$(kubectl get pods --ignore-not-found 2>&1 "${kube_flags[@]}")
  # Post-condition: The text "No resources found" should not be part of the output
  kube::test::if_has_not_string "${output_message}" 'No resources found'
  # Command
  output_message=$(kubectl get pods 2>&1 "${kube_flags[@]}" -o wide)
  # Post-condition: The text "No resources found" should be part of the output
  kube::test::if_has_string "${output_message}" 'No resources found'

  ### Test retrieval of non-existing POD with json output flag specified
  # Pre-condition: no POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  output_message=$(! kubectl get pods abc 2>&1 "${kube_flags[@]}" -o json)
  # Post-condition: POD abc should error since it doesn't exist
  kube::test::if_has_string "${output_message}" 'pods "abc" not found'
  # Post-condition: make sure we don't display an empty List
  if kube::test::if_has_string "${output_message}" 'List'; then
    echo 'Unexpected List output'
    echo "${LINENO} $(basename $0)"
    exit 1
  fi

  ### Test kubectl get all
  output_message=$(kubectl --v=6 --namespace default get all 2>&1 "${kube_flags[@]}")
  # Post-condition: Check if we get 200 OK from all the url(s)
  kube::test::if_has_string "${output_message}" "/api/v1/namespaces/default/pods 200 OK"
  kube::test::if_has_string "${output_message}" "/api/v1/namespaces/default/replicationcontrollers 200 OK"
  kube::test::if_has_string "${output_message}" "/api/v1/namespaces/default/services 200 OK"
  kube::test::if_has_string "${output_message}" "/apis/apps/v1beta1/namespaces/default/statefulsets 200 OK"
  kube::test::if_has_string "${output_message}" "/apis/autoscaling/v1/namespaces/default/horizontalpodautoscalers 200"
  kube::test::if_has_string "${output_message}" "/apis/batch/v1/namespaces/default/jobs 200 OK"
  kube::test::if_has_string "${output_message}" "/apis/extensions/v1beta1/namespaces/default/deployments 200 OK"
  kube::test::if_has_string "${output_message}" "/apis/extensions/v1beta1/namespaces/default/replicasets 200 OK"

  ### Test --allow-missing-template-keys
  # Pre-condition: no POD exists
  create_and_use_new_namespace
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml "${kube_flags[@]}"
  # Post-condition: valid-pod POD is created
  kubectl get "${kube_flags[@]}" pods -o json
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'

  ## check --allow-missing-template-keys defaults to true for jsonpath templates
  kubectl get "${kube_flags[@]}" pod valid-pod -o jsonpath='{.missing}'

  ## check --allow-missing-template-keys defaults to true for go templates
  kubectl get "${kube_flags[@]}" pod valid-pod -o go-template='{{.missing}}'

  ## check --allow-missing-template-keys=false results in an error for a missing key with jsonpath
  output_message=$(! kubectl get pod valid-pod --allow-missing-template-keys=false -o jsonpath='{.missing}' "${kube_flags[@]}")
  kube::test::if_has_string "${output_message}" 'missing is not found'

  ## check --allow-missing-template-keys=false results in an error for a missing key with go
  output_message=$(! kubectl get pod valid-pod --allow-missing-template-keys=false -o go-template='{{.missing}}' "${kube_flags[@]}")
  kube::test::if_has_string "${output_message}" 'map has no entry for key "missing"'

  ### Test kubectl get watch
  output_message=$(kubectl get pods -w --request-timeout=1 "${kube_flags[@]}")
  kube::test::if_has_string "${output_message}" 'STATUS'    # headers
  kube::test::if_has_string "${output_message}" 'valid-pod' # pod details
  output_message=$(kubectl get pods/valid-pod -o name -w --request-timeout=1 "${kube_flags[@]}")
  kube::test::if_has_not_string "${output_message}" 'STATUS' # no headers
  kube::test::if_has_string     "${output_message}" 'pods/valid-pod' # resource name
  output_message=$(kubectl get pods/valid-pod -o yaml -w --request-timeout=1 "${kube_flags[@]}")
  kube::test::if_has_not_string "${output_message}" 'STATUS'          # no headers
  kube::test::if_has_string     "${output_message}" 'name: valid-pod' # yaml
  output_message=$(! kubectl get pods/invalid-pod -w --request-timeout=1 "${kube_flags[@]}" 2>&1)
  kube::test::if_has_string "${output_message}" '"invalid-pod" not found'

  # cleanup
  kubectl delete pods valid-pod "${kube_flags[@]}"

  ### Test 'kubectl get -f <file> -o <non default printer>' prints all the items in the file's list
  # Pre-condition: no POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f test/fixtures/doc-yaml/user-guide/multi-pod.yaml "${kube_flags[@]}"
  # Post-condition: PODs redis-master and valid-pod exist

  # Check that all items in the list are printed
  output_message=$(kubectl get -f test/fixtures/doc-yaml/user-guide/multi-pod.yaml -o jsonpath="{..metadata.name}" "${kube_flags[@]}")
  kube::test::if_has_string "${output_message}" "redis-master valid-pod"

  # cleanup
  kubectl delete pods redis-master valid-pod "${kube_flags[@]}"

  set +o nounset
  set +o errexit
}

run_kubectl_request_timeout_tests() {
  set -o nounset
  set -o errexit

  kube::log::status "Testing kubectl request timeout"
  ### Test global request timeout option
  # Pre-condition: no POD exists
  create_and_use_new_namespace
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create "${kube_flags[@]}" -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml
  # Post-condition: valid-pod POD is created
  kubectl get "${kube_flags[@]}" pods -o json
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'

  ## check --request-timeout on 'get pod'
  output_message=$(kubectl get pod valid-pod --request-timeout=1)
  kube::test::if_has_string "${output_message}" 'valid-pod'

  ## check --request-timeout on 'get pod' with --watch
  output_message=$(kubectl get pod valid-pod --request-timeout=1 --watch 2>&1)
  kube::test::if_has_string "${output_message}" 'Timeout exceeded while reading body'

  ## check --request-timeout value with no time unit
  output_message=$(kubectl get pod valid-pod --request-timeout=1 2>&1)
  kube::test::if_has_string "${output_message}" 'valid-pod'

  ## check --request-timeout value with invalid time unit
  output_message=$(! kubectl get pod valid-pod --request-timeout="1p" 2>&1)
  kube::test::if_has_string "${output_message}" 'Invalid timeout value'

  # cleanup
  kubectl delete pods valid-pod "${kube_flags[@]}"

  set +o nounset
  set +o errexit
}

run_crd_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl crd"
  kubectl "${kube_flags_with_token[@]}" create -f - << __EOF__
{
  "kind": "CustomResourceDefinition",
  "apiVersion": "apiextensions.k8s.io/v1beta1",
  "metadata": {
    "name": "foos.company.com"
  },
  "spec": {
    "group": "company.com",
    "version": "v1",
    "names": {
      "plural": "foos",
      "kind": "Foo"
    }
  }
}
__EOF__

  # Post-Condition: assertion object exist
  kube::test::get_object_assert customresourcedefinitions "{{range.items}}{{$id_field}}:{{end}}" 'foos.company.com:'

  kubectl "${kube_flags_with_token[@]}" create -f - << __EOF__
{
  "kind": "CustomResourceDefinition",
  "apiVersion": "apiextensions.k8s.io/v1beta1",
  "metadata": {
    "name": "bars.company.com"
  },
  "spec": {
    "group": "company.com",
    "version": "v1",
    "names": {
      "plural": "bars",
      "kind": "Bar"
    }
  }
}
__EOF__

  # Post-Condition: assertion object exist
  kube::test::get_object_assert customresourcedefinitions "{{range.items}}{{$id_field}}:{{end}}" 'bars.company.com:foos.company.com:'

  run_non_native_resource_tests

  # teardown
  kubectl delete customresourcedefinitions/foos.company.com "${kube_flags_with_token[@]}"
  kubectl delete customresourcedefinitions/bars.company.com "${kube_flags_with_token[@]}"

  set +o nounset
  set +o errexit
}

kube::util::non_native_resources() {
  local times
  local wait
  local failed
  times=30
  wait=10
  local i
  for i in $(seq 1 $times); do
    failed=""
    kubectl "${kube_flags[@]}" get --raw '/apis/company.com/v1' || failed=true
    kubectl "${kube_flags[@]}" get --raw '/apis/company.com/v1/foos' || failed=true
    kubectl "${kube_flags[@]}" get --raw '/apis/company.com/v1/bars' || failed=true

    if [ -z "${failed}" ]; then
      return 0
    fi
    sleep ${wait}
  done

  kube::log::error "Timed out waiting for non-native-resources; tried ${times} waiting ${wait}s between each"
  return 1
}

run_non_native_resource_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl non-native resources"
  kube::util::non_native_resources

  # Test that we can list this new third party resource (foos)
  kube::test::get_object_assert foos "{{range.items}}{{$id_field}}:{{end}}" ''

  # Test that we can list this new third party resource (bars)
  kube::test::get_object_assert bars "{{range.items}}{{$id_field}}:{{end}}" ''

  # Test that we can create a new resource of type Foo
  kubectl "${kube_flags[@]}" create -f hack/testdata/TPR/foo.yaml "${kube_flags[@]}"

  # Test that we can list this new third party resource
  kube::test::get_object_assert foos "{{range.items}}{{$id_field}}:{{end}}" 'test:'

  # Test alternate forms
  kube::test::get_object_assert foo                 "{{range.items}}{{$id_field}}:{{end}}" 'test:'
  kube::test::get_object_assert foos.company.com    "{{range.items}}{{$id_field}}:{{end}}" 'test:'
  kube::test::get_object_assert foos.v1.company.com "{{range.items}}{{$id_field}}:{{end}}" 'test:'

  # Test all printers, with lists and individual items
  kube::log::status "Testing ThirdPartyResource printing"
  kubectl "${kube_flags[@]}" get foos
  kubectl "${kube_flags[@]}" get foos/test
  kubectl "${kube_flags[@]}" get foos      -o name
  kubectl "${kube_flags[@]}" get foos/test -o name
  kubectl "${kube_flags[@]}" get foos      -o wide
  kubectl "${kube_flags[@]}" get foos/test -o wide
  kubectl "${kube_flags[@]}" get foos      -o json
  kubectl "${kube_flags[@]}" get foos/test -o json
  kubectl "${kube_flags[@]}" get foos      -o yaml
  kubectl "${kube_flags[@]}" get foos/test -o yaml
  kubectl "${kube_flags[@]}" get foos      -o "jsonpath={.items[*].someField}" --allow-missing-template-keys=false
  kubectl "${kube_flags[@]}" get foos/test -o "jsonpath={.someField}"          --allow-missing-template-keys=false
  kubectl "${kube_flags[@]}" get foos      -o "go-template={{range .items}}{{.someField}}{{end}}" --allow-missing-template-keys=false
  kubectl "${kube_flags[@]}" get foos/test -o "go-template={{.someField}}"                        --allow-missing-template-keys=false
  output_message=$(kubectl "${kube_flags[@]}" get foos/test -o name)
  kube::test::if_has_string "${output_message}" 'foos/test'

  # Test patching
  kube::log::status "Testing ThirdPartyResource patching"
  kubectl "${kube_flags[@]}" patch foos/test -p '{"patched":"value1"}' --type=merge
  kube::test::get_object_assert foos/test "{{.patched}}" 'value1'
  kubectl "${kube_flags[@]}" patch foos/test -p '{"patched":"value2"}' --type=merge --record
  kube::test::get_object_assert foos/test "{{.patched}}" 'value2'
  kubectl "${kube_flags[@]}" patch foos/test -p '{"patched":null}' --type=merge --record
  kube::test::get_object_assert foos/test "{{.patched}}" '<no value>'
  # Get local version
  TPR_RESOURCE_FILE="${KUBE_TEMP}/tpr-foos-test.json"
  kubectl "${kube_flags[@]}" get foos/test -o json > "${TPR_RESOURCE_FILE}"
  # cannot apply strategic patch locally
  TPR_PATCH_ERROR_FILE="${KUBE_TEMP}/tpr-foos-test-error"
  ! kubectl "${kube_flags[@]}" patch --local -f "${TPR_RESOURCE_FILE}" -p '{"patched":"value3"}' 2> "${TPR_PATCH_ERROR_FILE}"
  if grep -q "try --type merge" "${TPR_PATCH_ERROR_FILE}"; then
    kube::log::status "\"kubectl patch --local\" returns error as expected for ThirdPartyResource: $(cat ${TPR_PATCH_ERROR_FILE})"
  else
    kube::log::status "\"kubectl patch --local\" returns unexpected error or non-error: $(cat ${TPR_PATCH_ERROR_FILE})"
    exit 1
  fi
  # can apply merge patch locally
  kubectl "${kube_flags[@]}" patch --local -f "${TPR_RESOURCE_FILE}" -p '{"patched":"value3"}' --type=merge -o json
  # can apply merge patch remotely
  kubectl "${kube_flags[@]}" patch --record -f "${TPR_RESOURCE_FILE}" -p '{"patched":"value3"}' --type=merge -o json
  kube::test::get_object_assert foos/test "{{.patched}}" 'value3'
  rm "${TPR_RESOURCE_FILE}"
  rm "${TPR_PATCH_ERROR_FILE}"

  # Test labeling
  kube::log::status "Testing ThirdPartyResource labeling"
  kubectl "${kube_flags[@]}" label foos --all listlabel=true
  kubectl "${kube_flags[@]}" label foo/test itemlabel=true

  # Test annotating
  kube::log::status "Testing ThirdPartyResource annotating"
  kubectl "${kube_flags[@]}" annotate foos --all listannotation=true
  kubectl "${kube_flags[@]}" annotate foo/test itemannotation=true

  # Test describing
  kube::log::status "Testing ThirdPartyResource describing"
  kubectl "${kube_flags[@]}" describe foos
  kubectl "${kube_flags[@]}" describe foos/test
  kubectl "${kube_flags[@]}" describe foos | grep listlabel=true
  kubectl "${kube_flags[@]}" describe foos | grep itemlabel=true

  # Delete the resource with cascade.
  kubectl "${kube_flags[@]}" delete foos test --cascade=true

  # Make sure it's gone
  kube::test::get_object_assert foos "{{range.items}}{{$id_field}}:{{end}}" ''

  # Test that we can create a new resource of type Bar
  kubectl "${kube_flags[@]}" create -f hack/testdata/TPR/bar.yaml "${kube_flags[@]}"

  # Test that we can list this new third party resource
  kube::test::get_object_assert bars "{{range.items}}{{$id_field}}:{{end}}" 'test:'

  # Test that we can watch the resource.
  # Start watcher in background with process substitution,
  # so we can read from stdout asynchronously.
  kube::log::status "Testing ThirdPartyResource watching"
  exec 3< <(kubectl "${kube_flags[@]}" get bars --request-timeout=1m --watch-only -o name & echo $! ; wait)
  local watch_pid
  read <&3 watch_pid

  # We can't be sure when the watch gets established,
  # so keep triggering events (in the background) until something comes through.
  local tries=0
  while [ ${tries} -lt 10 ]; do
    tries=$((tries+1))
    kubectl "${kube_flags[@]}" patch bars/test -p "{\"patched\":\"${tries}\"}" --type=merge
    sleep 1
  done &
  local patch_pid=$!

  # Wait up to 30s for a complete line of output.
  local watch_output
  read <&3 -t 30 watch_output
  # Stop the watcher and the patch loop.
  kill -9 ${watch_pid}
  kill -9 ${patch_pid}
  kube::test::if_has_string "${watch_output}" 'bars/test'

  # Delete the resource without cascade.
  kubectl "${kube_flags[@]}" delete bars test --cascade=false

  # Make sure it's gone
  kube::test::wait_object_assert bars "{{range.items}}{{$id_field}}:{{end}}" ''

  # Test that we can create single item via apply
  kubectl "${kube_flags[@]}" apply -f hack/testdata/TPR/foo.yaml

  # Test that we have create a foo named test
  kube::test::get_object_assert foos "{{range.items}}{{$id_field}}:{{end}}" 'test:'

  # Test that the field has the expected value
  kube::test::get_object_assert foos/test '{{.someField}}' 'field1'

  # Test that apply an empty patch doesn't change fields
  kubectl "${kube_flags[@]}" apply -f hack/testdata/TPR/foo.yaml

  # Test that the field has the same value after re-apply
  kube::test::get_object_assert foos/test '{{.someField}}' 'field1'

  # Test that apply has updated the subfield
  kube::test::get_object_assert foos/test '{{.nestedField.someSubfield}}' 'subfield1'

  # Update a subfield and then apply the change
  kubectl "${kube_flags[@]}" apply -f hack/testdata/TPR/foo-updated-subfield.yaml

  # Test that apply has updated the subfield
  kube::test::get_object_assert foos/test '{{.nestedField.someSubfield}}' 'modifiedSubfield'

  # Test that the field has the expected value
  kube::test::get_object_assert foos/test '{{.nestedField.otherSubfield}}' 'subfield2'

  # Delete a subfield and then apply the change
  kubectl "${kube_flags[@]}" apply -f hack/testdata/TPR/foo-deleted-subfield.yaml

  # Test that apply has deleted the field
  kube::test::get_object_assert foos/test '{{.nestedField.otherSubfield}}' '<no value>'

  # Test that the field does not exist
  kube::test::get_object_assert foos/test '{{.nestedField.newSubfield}}' '<no value>'

  # Add a field and then apply the change
  kubectl "${kube_flags[@]}" apply -f hack/testdata/TPR/foo-added-subfield.yaml

  # Test that apply has added the field
  kube::test::get_object_assert foos/test '{{.nestedField.newSubfield}}' 'subfield3'

  # Delete the resource
  kubectl "${kube_flags[@]}" delete -f hack/testdata/TPR/foo.yaml

  # Make sure it's gone
  kube::test::get_object_assert foos "{{range.items}}{{$id_field}}:{{end}}" ''

  # Test that we can create list via apply
  kubectl "${kube_flags[@]}" apply -f hack/testdata/TPR/multi-tpr-list.yaml

  # Test that we have create a foo and a bar from a list
  kube::test::get_object_assert foos "{{range.items}}{{$id_field}}:{{end}}" 'test-list:'
  kube::test::get_object_assert bars "{{range.items}}{{$id_field}}:{{end}}" 'test-list:'

  # Test that the field has the expected value
  kube::test::get_object_assert foos/test-list '{{.someField}}' 'field1'
  kube::test::get_object_assert bars/test-list '{{.someField}}' 'field1'

  # Test that re-apply an list doesn't change anything
  kubectl "${kube_flags[@]}" apply -f hack/testdata/TPR/multi-tpr-list.yaml

  # Test that the field has the same value after re-apply
  kube::test::get_object_assert foos/test-list '{{.someField}}' 'field1'
  kube::test::get_object_assert bars/test-list '{{.someField}}' 'field1'

  # Test that the fields have the expected value
  kube::test::get_object_assert foos/test-list '{{.someField}}' 'field1'
  kube::test::get_object_assert bars/test-list '{{.someField}}' 'field1'

  # Update fields and then apply the change
  kubectl "${kube_flags[@]}" apply -f hack/testdata/TPR/multi-tpr-list-updated-field.yaml

  # Test that apply has updated the fields
  kube::test::get_object_assert foos/test-list '{{.someField}}' 'modifiedField'
  kube::test::get_object_assert bars/test-list '{{.someField}}' 'modifiedField'

  # Test that the field has the expected value
  kube::test::get_object_assert foos/test-list '{{.otherField}}' 'field2'
  kube::test::get_object_assert bars/test-list '{{.otherField}}' 'field2'

  # Delete fields and then apply the change
  kubectl "${kube_flags[@]}" apply -f hack/testdata/TPR/multi-tpr-list-deleted-field.yaml

  # Test that apply has deleted the fields
  kube::test::get_object_assert foos/test-list '{{.otherField}}' '<no value>'
  kube::test::get_object_assert bars/test-list '{{.otherField}}' '<no value>'

  # Test that the fields does not exist
  kube::test::get_object_assert foos/test-list '{{.newField}}' '<no value>'
  kube::test::get_object_assert bars/test-list '{{.newField}}' '<no value>'

  # Add a field and then apply the change
  kubectl "${kube_flags[@]}" apply -f hack/testdata/TPR/multi-tpr-list-added-field.yaml

  # Test that apply has added the field
  kube::test::get_object_assert foos/test-list '{{.newField}}' 'field3'
  kube::test::get_object_assert bars/test-list '{{.newField}}' 'field3'

  # Delete the resource
  kubectl "${kube_flags[@]}" delete -f hack/testdata/TPR/multi-tpr-list.yaml

  # Make sure it's gone
  kube::test::get_object_assert foos "{{range.items}}{{$id_field}}:{{end}}" ''
  kube::test::get_object_assert bars "{{range.items}}{{$id_field}}:{{end}}" ''

  ## kubectl apply --prune
  # Test that no foo or bar exist
  kube::test::get_object_assert foos "{{range.items}}{{$id_field}}:{{end}}" ''
  kube::test::get_object_assert bars "{{range.items}}{{$id_field}}:{{end}}" ''

  # apply --prune on foo.yaml that has foo/test
  kubectl apply --prune -l pruneGroup=true -f hack/testdata/TPR/foo.yaml "${kube_flags[@]}" --prune-whitelist=company.com/v1/Foo --prune-whitelist=company.com/v1/Bar
  # check right tprs exist
  kube::test::get_object_assert foos "{{range.items}}{{$id_field}}:{{end}}" 'test:'
  kube::test::get_object_assert bars "{{range.items}}{{$id_field}}:{{end}}" ''

  # apply --prune on bar.yaml that has bar/test
  kubectl apply --prune -l pruneGroup=true -f hack/testdata/TPR/bar.yaml "${kube_flags[@]}" --prune-whitelist=company.com/v1/Foo --prune-whitelist=company.com/v1/Bar
  # check right tprs exist
  kube::test::get_object_assert foos "{{range.items}}{{$id_field}}:{{end}}" ''
  kube::test::get_object_assert bars "{{range.items}}{{$id_field}}:{{end}}" 'test:'

  # Delete the resource
  kubectl "${kube_flags[@]}" delete -f hack/testdata/TPR/bar.yaml

  # Make sure it's gone
  kube::test::get_object_assert foos "{{range.items}}{{$id_field}}:{{end}}" ''
  kube::test::get_object_assert bars "{{range.items}}{{$id_field}}:{{end}}" ''

  # Test 'kubectl create' with namespace, and namespace cleanup.
  kubectl "${kube_flags[@]}" create namespace non-native-resources
  kubectl "${kube_flags[@]}" create -f hack/testdata/TPR/bar.yaml --namespace=non-native-resources
  kube::test::get_object_assert bars '{{len .items}}' '1' --namespace=non-native-resources
  kubectl "${kube_flags[@]}" delete namespace non-native-resources
  # Make sure objects go away.
  kube::test::wait_object_assert bars '{{len .items}}' '0' --namespace=non-native-resources
  # Make sure namespace goes away.
  local tries=0
  while kubectl "${kube_flags[@]}" get namespace non-native-resources && [ ${tries} -lt 10 ]; do
    tries=$((tries+1))
    sleep ${tries}
  done

  set +o nounset
  set +o errexit
}

run_recursive_resources_tests() {
  set -o nounset
  set -o errexit

  kube::log::status "Testing recursive resources"
  ### Create multiple busybox PODs recursively from directory of YAML files
  # Pre-condition: no POD exists
  create_and_use_new_namespace
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  output_message=$(! kubectl create -f hack/testdata/recursive/pod --recursive 2>&1 "${kube_flags[@]}")
  # Post-condition: busybox0 & busybox1 PODs are created, and since busybox2 is malformed, it should error
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'busybox0:busybox1:'
  kube::test::if_has_string "${output_message}" 'error validating data: kind not set'

  ## Edit multiple busybox PODs by updating the image field of multiple PODs recursively from a directory. tmp-editor.sh is a fake editor
  # Pre-condition: busybox0 & busybox1 PODs exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'busybox0:busybox1:'
  # Command
  echo -e '#!/bin/bash\nsed -i "s/image: busybox/image: prom\/busybox/g" $1' > /tmp/tmp-editor.sh
  chmod +x /tmp/tmp-editor.sh
  output_message=$(! EDITOR=/tmp/tmp-editor.sh kubectl edit -f hack/testdata/recursive/pod --recursive 2>&1 "${kube_flags[@]}")
  # Post-condition: busybox0 & busybox1 PODs are not edited, and since busybox2 is malformed, it should error
  # The reason why busybox0 & busybox1 PODs are not edited is because the editor tries to load all objects in
  # a list but since it contains invalid objects, it will never open.
  kube::test::get_object_assert pods "{{range.items}}{{$image_field}}:{{end}}" 'busybox:busybox:'
  kube::test::if_has_string "${output_message}" "Object 'Kind' is missing"
  # cleaning
  rm /tmp/tmp-editor.sh

  ## Replace multiple busybox PODs recursively from directory of YAML files
  # Pre-condition: busybox0 & busybox1 PODs exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'busybox0:busybox1:'
  # Command
  output_message=$(! kubectl replace -f hack/testdata/recursive/pod-modify --recursive 2>&1 "${kube_flags[@]}")
  # Post-condition: busybox0 & busybox1 PODs are replaced, and since busybox2 is malformed, it should error
  kube::test::get_object_assert pods "{{range.items}}{{${labels_field}.status}}:{{end}}" 'replaced:replaced:'
  kube::test::if_has_string "${output_message}" 'error validating data: kind not set'

  ## Describe multiple busybox PODs recursively from directory of YAML files
  # Pre-condition: busybox0 & busybox1 PODs exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'busybox0:busybox1:'
  # Command
  output_message=$(! kubectl describe -f hack/testdata/recursive/pod --recursive 2>&1 "${kube_flags[@]}")
  # Post-condition: busybox0 & busybox1 PODs are described, and since busybox2 is malformed, it should error
  kube::test::if_has_string "${output_message}" "app=busybox0"
  kube::test::if_has_string "${output_message}" "app=busybox1"
  kube::test::if_has_string "${output_message}" "Object 'Kind' is missing"

  ## Annotate multiple busybox PODs recursively from directory of YAML files
  # Pre-condition: busybox0 & busybox1 PODs exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'busybox0:busybox1:'
  # Command
  output_message=$(! kubectl annotate -f hack/testdata/recursive/pod annotatekey='annotatevalue' --recursive 2>&1 "${kube_flags[@]}")
  # Post-condition: busybox0 & busybox1 PODs are annotated, and since busybox2 is malformed, it should error
  kube::test::get_object_assert pods "{{range.items}}{{${annotations_field}.annotatekey}}:{{end}}" 'annotatevalue:annotatevalue:'
  kube::test::if_has_string "${output_message}" "Object 'Kind' is missing"

  ## Apply multiple busybox PODs recursively from directory of YAML files
  # Pre-condition: busybox0 & busybox1 PODs exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'busybox0:busybox1:'
  # Command
  output_message=$(! kubectl apply -f hack/testdata/recursive/pod-modify --recursive 2>&1 "${kube_flags[@]}")
  # Post-condition: busybox0 & busybox1 PODs are updated, and since busybox2 is malformed, it should error
  kube::test::get_object_assert pods "{{range.items}}{{${labels_field}.status}}:{{end}}" 'replaced:replaced:'
  kube::test::if_has_string "${output_message}" 'error validating data: kind not set'


  ### Convert deployment YAML file locally without affecting the live deployment.
  # Pre-condition: no deployments exist
  kube::test::get_object_assert deployment "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  # Create a deployment (revision 1)
  kubectl create -f hack/testdata/deployment-revision1.yaml "${kube_flags[@]}"
  kube::test::get_object_assert deployment "{{range.items}}{{$id_field}}:{{end}}" 'nginx:'
  kube::test::get_object_assert deployment "{{range.items}}{{$image_field0}}:{{end}}" "${IMAGE_DEPLOYMENT_R1}:"
  # Command
  output_message=$(kubectl convert --local -f hack/testdata/deployment-revision1.yaml --output-version=apps/v1beta1 -o go-template='{{ .apiVersion }}' "${kube_flags[@]}")
  echo $output_message
  # Post-condition: apiVersion is still extensions/v1beta1 in the live deployment, but command output is the new value
  kube::test::get_object_assert 'deployment nginx' "{{ .apiVersion }}" 'extensions/v1beta1'
  kube::test::if_has_string "${output_message}" "apps/v1beta1"
  # Clean up
  kubectl delete deployment nginx "${kube_flags[@]}"

  ## Convert multiple busybox PODs recursively from directory of YAML files
  # Pre-condition: busybox0 & busybox1 PODs exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'busybox0:busybox1:'
  # Command
  output_message=$(! kubectl convert -f hack/testdata/recursive/pod --recursive 2>&1 "${kube_flags[@]}")
  # Post-condition: busybox0 & busybox1 PODs are converted, and since busybox2 is malformed, it should error
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'busybox0:busybox1:'
  kube::test::if_has_string "${output_message}" "Object 'Kind' is missing"

  ## Get multiple busybox PODs recursively from directory of YAML files
  # Pre-condition: busybox0 & busybox1 PODs exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'busybox0:busybox1:'
  # Command
  output_message=$(! kubectl get -f hack/testdata/recursive/pod --recursive 2>&1 "${kube_flags[@]}" -o go-template="{{range.items}}{{$id_field}}:{{end}}")
  # Post-condition: busybox0 & busybox1 PODs are retrieved, but because busybox2 is malformed, it should not show up
  kube::test::if_has_string "${output_message}" "busybox0:busybox1:"
  kube::test::if_has_string "${output_message}" "Object 'Kind' is missing"

  ## Label multiple busybox PODs recursively from directory of YAML files
  # Pre-condition: busybox0 & busybox1 PODs exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'busybox0:busybox1:'
  # Command
  output_message=$(! kubectl label -f hack/testdata/recursive/pod mylabel='myvalue' --recursive 2>&1 "${kube_flags[@]}")
  echo $output_message
  # Post-condition: busybox0 & busybox1 PODs are labeled, but because busybox2 is malformed, it should not show up
  kube::test::get_object_assert pods "{{range.items}}{{${labels_field}.mylabel}}:{{end}}" 'myvalue:myvalue:'
  kube::test::if_has_string "${output_message}" "Object 'Kind' is missing"

  ## Patch multiple busybox PODs recursively from directory of YAML files
  # Pre-condition: busybox0 & busybox1 PODs exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'busybox0:busybox1:'
  # Command
  output_message=$(! kubectl patch -f hack/testdata/recursive/pod -p='{"spec":{"containers":[{"name":"busybox","image":"prom/busybox"}]}}' --recursive 2>&1 "${kube_flags[@]}")
  echo $output_message
  # Post-condition: busybox0 & busybox1 PODs are patched, but because busybox2 is malformed, it should not show up
  kube::test::get_object_assert pods "{{range.items}}{{$image_field}}:{{end}}" 'prom/busybox:prom/busybox:'
  kube::test::if_has_string "${output_message}" "Object 'Kind' is missing"

  ### Delete multiple busybox PODs recursively from directory of YAML files
  # Pre-condition: busybox0 & busybox1 PODs exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'busybox0:busybox1:'
  # Command
  output_message=$(! kubectl delete -f hack/testdata/recursive/pod --recursive --grace-period=0 --force 2>&1 "${kube_flags[@]}")
  # Post-condition: busybox0 & busybox1 PODs are deleted, and since busybox2 is malformed, it should error
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  kube::test::if_has_string "${output_message}" "Object 'Kind' is missing"

  ### Create replication controller recursively from directory of YAML files
  # Pre-condition: no replication controller exists
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  ! kubectl create -f hack/testdata/recursive/rc --recursive "${kube_flags[@]}"
  # Post-condition: frontend replication controller is created
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" 'busybox0:busybox1:'

  ### Autoscale multiple replication controllers recursively from directory of YAML files
  # Pre-condition: busybox0 & busybox1 replication controllers exist & 1
  # replica each
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" 'busybox0:busybox1:'
  kube::test::get_object_assert 'rc busybox0' "{{$rc_replicas_field}}" '1'
  kube::test::get_object_assert 'rc busybox1' "{{$rc_replicas_field}}" '1'
  # Command
  output_message=$(! kubectl autoscale --min=1 --max=2 -f hack/testdata/recursive/rc --recursive 2>&1 "${kube_flags[@]}")
  # Post-condition: busybox0 & busybox replication controllers are autoscaled
  # with min. of 1 replica & max of 2 replicas, and since busybox2 is malformed, it should error
  kube::test::get_object_assert 'hpa busybox0' "{{$hpa_min_field}} {{$hpa_max_field}} {{$hpa_cpu_field}}" '1 2 80'
  kube::test::get_object_assert 'hpa busybox1' "{{$hpa_min_field}} {{$hpa_max_field}} {{$hpa_cpu_field}}" '1 2 80'
  kube::test::if_has_string "${output_message}" "Object 'Kind' is missing"
  kubectl delete hpa busybox0 "${kube_flags[@]}"
  kubectl delete hpa busybox1 "${kube_flags[@]}"

  ### Expose multiple replication controllers as service recursively from directory of YAML files
  # Pre-condition: busybox0 & busybox1 replication controllers exist & 1
  # replica each
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" 'busybox0:busybox1:'
  kube::test::get_object_assert 'rc busybox0' "{{$rc_replicas_field}}" '1'
  kube::test::get_object_assert 'rc busybox1' "{{$rc_replicas_field}}" '1'
  # Command
  output_message=$(! kubectl expose -f hack/testdata/recursive/rc --recursive --port=80 2>&1 "${kube_flags[@]}")
  # Post-condition: service exists and the port is unnamed
  kube::test::get_object_assert 'service busybox0' "{{$port_name}} {{$port_field}}" '<no value> 80'
  kube::test::get_object_assert 'service busybox1' "{{$port_name}} {{$port_field}}" '<no value> 80'
  kube::test::if_has_string "${output_message}" "Object 'Kind' is missing"

  ### Scale multiple replication controllers recursively from directory of YAML files
  # Pre-condition: busybox0 & busybox1 replication controllers exist & 1
  # replica each
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" 'busybox0:busybox1:'
  kube::test::get_object_assert 'rc busybox0' "{{$rc_replicas_field}}" '1'
  kube::test::get_object_assert 'rc busybox1' "{{$rc_replicas_field}}" '1'
  # Command
  output_message=$(! kubectl scale --current-replicas=1 --replicas=2 -f hack/testdata/recursive/rc --recursive 2>&1 "${kube_flags[@]}")
  # Post-condition: busybox0 & busybox1 replication controllers are scaled to 2 replicas, and since busybox2 is malformed, it should error
  kube::test::get_object_assert 'rc busybox0' "{{$rc_replicas_field}}" '2'
  kube::test::get_object_assert 'rc busybox1' "{{$rc_replicas_field}}" '2'
  kube::test::if_has_string "${output_message}" "Object 'Kind' is missing"

  ### Delete multiple busybox replication controllers recursively from directory of YAML files
  # Pre-condition: busybox0 & busybox1 PODs exist
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" 'busybox0:busybox1:'
  # Command
  output_message=$(! kubectl delete -f hack/testdata/recursive/rc --recursive --grace-period=0 --force 2>&1 "${kube_flags[@]}")
  # Post-condition: busybox0 & busybox1 replication controllers are deleted, and since busybox2 is malformed, it should error
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  kube::test::if_has_string "${output_message}" "Object 'Kind' is missing"

  ### Rollout on multiple deployments recursively
  # Pre-condition: no deployments exist
  kube::test::get_object_assert deployment "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  # Create deployments (revision 1) recursively from directory of YAML files
  ! kubectl create -f hack/testdata/recursive/deployment --recursive "${kube_flags[@]}"
  kube::test::get_object_assert deployment "{{range.items}}{{$id_field}}:{{end}}" 'nginx0-deployment:nginx1-deployment:'
  kube::test::get_object_assert deployment "{{range.items}}{{$image_field0}}:{{end}}" "${IMAGE_NGINX}:${IMAGE_NGINX}:"
  ## Rollback the deployments to revision 1 recursively
  output_message=$(! kubectl rollout undo -f hack/testdata/recursive/deployment --recursive --to-revision=1 2>&1 "${kube_flags[@]}")
  # Post-condition: nginx0 & nginx1 should be a no-op, and since nginx2 is malformed, it should error
  kube::test::get_object_assert deployment "{{range.items}}{{$image_field0}}:{{end}}" "${IMAGE_NGINX}:${IMAGE_NGINX}:"
  kube::test::if_has_string "${output_message}" "Object 'Kind' is missing"
  ## Pause the deployments recursively
  PRESERVE_ERR_FILE=true
  kubectl-with-retry rollout pause -f hack/testdata/recursive/deployment --recursive "${kube_flags[@]}"
  output_message=$(cat ${ERROR_FILE})
  # Post-condition: nginx0 & nginx1 should both have paused set to true, and since nginx2 is malformed, it should error
  kube::test::get_object_assert deployment "{{range.items}}{{.spec.paused}}:{{end}}" "true:true:"
  kube::test::if_has_string "${output_message}" "Object 'Kind' is missing"
  ## Resume the deployments recursively
  kubectl-with-retry rollout resume -f hack/testdata/recursive/deployment --recursive "${kube_flags[@]}"
  output_message=$(cat ${ERROR_FILE})
  # Post-condition: nginx0 & nginx1 should both have paused set to nothing, and since nginx2 is malformed, it should error
  kube::test::get_object_assert deployment "{{range.items}}{{.spec.paused}}:{{end}}" "<no value>:<no value>:"
  kube::test::if_has_string "${output_message}" "Object 'Kind' is missing"
  ## Retrieve the rollout history of the deployments recursively
  output_message=$(! kubectl rollout history -f hack/testdata/recursive/deployment --recursive 2>&1 "${kube_flags[@]}")
  # Post-condition: nginx0 & nginx1 should both have a history, and since nginx2 is malformed, it should error
  kube::test::if_has_string "${output_message}" "nginx0-deployment"
  kube::test::if_has_string "${output_message}" "nginx1-deployment"
  kube::test::if_has_string "${output_message}" "Object 'Kind' is missing"
  # Clean up
  unset PRESERVE_ERR_FILE
  rm "${ERROR_FILE}"
  ! kubectl delete -f hack/testdata/recursive/deployment --recursive "${kube_flags[@]}" --grace-period=0 --force
  sleep 1

  ### Rollout on multiple replication controllers recursively - these tests ensure that rollouts cannot be performed on resources that don't support it
  # Pre-condition: no replication controller exists
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  # Create replication controllers recursively from directory of YAML files
  ! kubectl create -f hack/testdata/recursive/rc --recursive "${kube_flags[@]}"
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" 'busybox0:busybox1:'
  # Command
  ## Attempt to rollback the replication controllers to revision 1 recursively
  output_message=$(! kubectl rollout undo -f hack/testdata/recursive/rc --recursive --to-revision=1 2>&1 "${kube_flags[@]}")
  # Post-condition: busybox0 & busybox1 should error as they are RC's, and since busybox2 is malformed, it should error
  kube::test::if_has_string "${output_message}" 'no rollbacker has been implemented for {"" "ReplicationController"}'
  kube::test::if_has_string "${output_message}" "Object 'Kind' is missing"
  ## Attempt to pause the replication controllers recursively
  output_message=$(! kubectl rollout pause -f hack/testdata/recursive/rc --recursive 2>&1 "${kube_flags[@]}")
  # Post-condition: busybox0 & busybox1 should error as they are RC's, and since busybox2 is malformed, it should error
  kube::test::if_has_string "${output_message}" "Object 'Kind' is missing"
  kube::test::if_has_string "${output_message}" 'replicationcontrollers "busybox0" pausing is not supported'
  kube::test::if_has_string "${output_message}" 'replicationcontrollers "busybox1" pausing is not supported'
  ## Attempt to resume the replication controllers recursively
  output_message=$(! kubectl rollout resume -f hack/testdata/recursive/rc --recursive 2>&1 "${kube_flags[@]}")
  # Post-condition: busybox0 & busybox1 should error as they are RC's, and since busybox2 is malformed, it should error
  kube::test::if_has_string "${output_message}" "Object 'Kind' is missing"
  kube::test::if_has_string "${output_message}" 'replicationcontrollers "busybox0" resuming is not supported'
  kube::test::if_has_string "${output_message}" 'replicationcontrollers "busybox0" resuming is not supported'
  # Clean up
  ! kubectl delete -f hack/testdata/recursive/rc --recursive "${kube_flags[@]}" --grace-period=0 --force
  sleep 1

  set +o nounset
  set +o errexit
}

run_namespace_tests() {
  set -o nounset
  set -o errexit

  kube::log::status "Testing kubectl(v1:namespaces)"
  ### Create a new namespace
  # Pre-condition: only the "default" namespace exists
  # The Pre-condition doesn't hold anymore after we create and switch namespaces before creating pods with same name in the test.
  # kube::test::get_object_assert namespaces "{{range.items}}{{$id_field}}:{{end}}" 'default:'
  # Command
  kubectl create namespace my-namespace
  # Post-condition: namespace 'my-namespace' is created.
  kube::test::get_object_assert 'namespaces/my-namespace' "{{$id_field}}" 'my-namespace'
  # Clean up
  kubectl delete namespace my-namespace

  ######################
  # Pods in Namespaces #
  ######################

  if kube::test::if_supports_resource "${pods}" ; then
    ### Create a new namespace
    # Pre-condition: the other namespace does not exist
    kube::test::get_object_assert 'namespaces' '{{range.items}}{{ if eq $id_field \"other\" }}found{{end}}{{end}}:' ':'
    # Command
    kubectl create namespace other
    # Post-condition: namespace 'other' is created.
    kube::test::get_object_assert 'namespaces/other' "{{$id_field}}" 'other'

    ### Create POD valid-pod in specific namespace
    # Pre-condition: no POD exists
    kube::test::get_object_assert 'pods --namespace=other' "{{range.items}}{{$id_field}}:{{end}}" ''
    # Command
    kubectl create "${kube_flags[@]}" --namespace=other -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml
    # Post-condition: valid-pod POD is created
    kube::test::get_object_assert 'pods --namespace=other' "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'
    # Post-condition: verify shorthand `-n other` has the same results as `--namespace=other`
    kube::test::get_object_assert 'pods -n other' "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'
    # Post-condition: a resource cannot be retrieved by name across all namespaces
    output_message=$(! kubectl get "${kube_flags[@]}" pod valid-pod --all-namespaces 2>&1)
    kube::test::if_has_string "${output_message}" "a resource cannot be retrieved by name across all namespaces"

    ### Delete POD valid-pod in specific namespace
    # Pre-condition: valid-pod POD exists
    kube::test::get_object_assert 'pods --namespace=other' "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'
    # Command
    kubectl delete "${kube_flags[@]}" pod --namespace=other valid-pod --grace-period=0 --force
    # Post-condition: valid-pod POD doesn't exist
    kube::test::get_object_assert 'pods --namespace=other' "{{range.items}}{{$id_field}}:{{end}}" ''
    # Clean up
    kubectl delete namespace other
  fi

  set +o nounset
  set +o errexit
}

run_secrets_test() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing secrets"
  ### Create a new namespace
  # Pre-condition: the test-secrets namespace does not exist
  kube::test::get_object_assert 'namespaces' '{{range.items}}{{ if eq $id_field \"test-secrets\" }}found{{end}}{{end}}:' ':'
  # Command
  kubectl create namespace test-secrets
  # Post-condition: namespace 'test-secrets' is created.
  kube::test::get_object_assert 'namespaces/test-secrets' "{{$id_field}}" 'test-secrets'

  ### Create a generic secret in a specific namespace
  # Pre-condition: no SECRET exists
  kube::test::get_object_assert 'secrets --namespace=test-secrets' "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create secret generic test-secret --from-literal=key1=value1 --type=test-type --namespace=test-secrets
  # Post-condition: secret exists and has expected values
  kube::test::get_object_assert 'secret/test-secret --namespace=test-secrets' "{{$id_field}}" 'test-secret'
  kube::test::get_object_assert 'secret/test-secret --namespace=test-secrets' "{{$secret_type}}" 'test-type'
  [[ "$(kubectl get secret/test-secret --namespace=test-secrets -o yaml "${kube_flags[@]}" | grep 'key1: dmFsdWUx')" ]]
  # Clean-up
  kubectl delete secret test-secret --namespace=test-secrets

  ### Create a docker-registry secret in a specific namespace
  if [[ "${WAIT_FOR_DELETION:-}" == "true" ]]; then
    kube::test::wait_object_assert 'secrets --namespace=test-secrets' "{{range.items}}{{$id_field}}:{{end}}" ''
  fi
  # Pre-condition: no SECRET exists
  kube::test::get_object_assert 'secrets --namespace=test-secrets' "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create secret docker-registry test-secret --docker-username=test-user --docker-password=test-password --docker-email='test-user@test.com' --namespace=test-secrets
  # Post-condition: secret exists and has expected values
  kube::test::get_object_assert 'secret/test-secret --namespace=test-secrets' "{{$id_field}}" 'test-secret'
  kube::test::get_object_assert 'secret/test-secret --namespace=test-secrets' "{{$secret_type}}" 'kubernetes.io/dockercfg'
  [[ "$(kubectl get secret/test-secret --namespace=test-secrets -o yaml "${kube_flags[@]}" | grep '.dockercfg:')" ]]
  # Clean-up
  kubectl delete secret test-secret --namespace=test-secrets

  ### Create a tls secret
  if [[ "${WAIT_FOR_DELETION:-}" == "true" ]]; then
    kube::test::wait_object_assert 'secrets --namespace=test-secrets' "{{range.items}}{{$id_field}}:{{end}}" ''
  fi
  # Pre-condition: no SECRET exists
  kube::test::get_object_assert 'secrets --namespace=test-secrets' "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create secret tls test-secret --namespace=test-secrets --key=hack/testdata/tls.key --cert=hack/testdata/tls.crt
  kube::test::get_object_assert 'secret/test-secret --namespace=test-secrets' "{{$id_field}}" 'test-secret'
  kube::test::get_object_assert 'secret/test-secret --namespace=test-secrets' "{{$secret_type}}" 'kubernetes.io/tls'
  # Clean-up
  kubectl delete secret test-secret --namespace=test-secrets

  # Create a secret using stringData
  kubectl create --namespace=test-secrets -f - "${kube_flags[@]}" << __EOF__
{
  "kind": "Secret",
  "apiVersion": "v1",
  "metadata": {
    "name": "secret-string-data"
  },
  "data": {
    "k1":"djE=",
    "k2":""
  },
  "stringData": {
    "k2":"v2"
  }
}
__EOF__
  # Post-condition: secret-string-data secret is created with expected data, merged/overridden data from stringData, and a cleared stringData field
  kube::test::get_object_assert 'secret/secret-string-data --namespace=test-secrets ' '{{.data}}' '.*k1:djE=.*'
  kube::test::get_object_assert 'secret/secret-string-data --namespace=test-secrets ' '{{.data}}' '.*k2:djI=.*'
  kube::test::get_object_assert 'secret/secret-string-data --namespace=test-secrets ' '{{.stringData}}' '<no value>'
  # Clean up
  kubectl delete secret secret-string-data --namespace=test-secrets

  ### Create a secret using output flags
  if [[ "${WAIT_FOR_DELETION:-}" == "true" ]]; then
    kube::test::wait_object_assert 'secrets --namespace=test-secrets' "{{range.items}}{{$id_field}}:{{end}}" ''
  fi
  # Pre-condition: no secret exists
  kube::test::get_object_assert 'secrets --namespace=test-secrets' "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  [[ "$(kubectl create secret generic test-secret --namespace=test-secrets --from-literal=key1=value1 --output=go-template --template=\"{{.metadata.name}}:\" | grep 'test-secret:')" ]]
  ## Clean-up
  kubectl delete secret test-secret --namespace=test-secrets
  # Clean up
  kubectl delete namespace test-secrets

  set +o nounset
  set +o errexit
}

run_configmap_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing configmaps"
  kubectl create -f test/fixtures/doc-yaml/user-guide/configmap/configmap.yaml
  kube::test::get_object_assert configmap "{{range.items}}{{$id_field}}{{end}}" 'test-configmap'
  kubectl delete configmap test-configmap "${kube_flags[@]}"

  ### Create a new namespace
  # Pre-condition: the test-configmaps namespace does not exist
  kube::test::get_object_assert 'namespaces' '{{range.items}}{{ if eq $id_field \"test-configmaps\" }}found{{end}}{{end}}:' ':'
  # Command
  kubectl create namespace test-configmaps
  # Post-condition: namespace 'test-configmaps' is created.
  kube::test::get_object_assert 'namespaces/test-configmaps' "{{$id_field}}" 'test-configmaps'

  ### Create a generic configmap in a specific namespace
  # Pre-condition: no configmaps namespace exists
  kube::test::get_object_assert 'configmaps --namespace=test-configmaps' "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create configmap test-configmap --from-literal=key1=value1 --namespace=test-configmaps
  # Post-condition: configmap exists and has expected values
  kube::test::get_object_assert 'configmap/test-configmap --namespace=test-configmaps' "{{$id_field}}" 'test-configmap'
  [[ "$(kubectl get configmap/test-configmap --namespace=test-configmaps -o yaml "${kube_flags[@]}" | grep 'key1: value1')" ]]
  # Clean-up
  kubectl delete configmap test-configmap --namespace=test-configmaps
  kubectl delete namespace test-configmaps

  set +o nounset
  set +o errexit
}

run_service_tests() {
  set -o nounset
  set -o errexit

  # switch back to the default namespace
  kubectl config set-context "${CONTEXT}" --namespace=""
  kube::log::status "Testing kubectl(v1:services)"

  ### Create redis-master service from JSON
  # Pre-condition: Only the default kubernetes services exist
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:'
  # Command
  kubectl create -f examples/guestbook/redis-master-service.yaml "${kube_flags[@]}"
  # Post-condition: redis-master service exists
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:redis-master:'
  # Describe command should print detailed information
  kube::test::describe_object_assert services 'redis-master' "Name:" "Labels:" "Selector:" "IP:" "Port:" "Endpoints:" "Session Affinity:"
  # Describe command should print events information by default
  kube::test::describe_object_events_assert services 'redis-master'
  # Describe command should not print events information when show-events=false
  kube::test::describe_object_events_assert services 'redis-master' false
  # Describe command should print events information when show-events=true
  kube::test::describe_object_events_assert services 'redis-master' true
  # Describe command (resource only) should print detailed information
  kube::test::describe_resource_assert services "Name:" "Labels:" "Selector:" "IP:" "Port:" "Endpoints:" "Session Affinity:"
  # Describe command should print events information by default
  kube::test::describe_resource_events_assert services
  # Describe command should not print events information when show-events=false
  kube::test::describe_resource_events_assert services false
  # Describe command should print events information when show-events=true
  kube::test::describe_resource_events_assert services true

  ### set selector
  # prove role=master
  kube::test::get_object_assert 'services redis-master' "{{range$service_selector_field}}{{.}}:{{end}}" "redis:master:backend:"

  # Set selector of a local file without talking to the server
  kubectl set selector -f examples/guestbook/redis-master-service.yaml role=padawan --local -o yaml "${kube_flags[@]}"
  ! kubectl set selector -f examples/guestbook/redis-master-service.yaml role=padawan --dry-run -o yaml "${kube_flags[@]}"
  # Set command to change the selector.
  kubectl set selector -f examples/guestbook/redis-master-service.yaml role=padawan
  # prove role=padawan
  kube::test::get_object_assert 'services redis-master' "{{range$service_selector_field}}{{.}}:{{end}}" "padawan:"
  # Set command to reset the selector back to the original one.
  kubectl set selector -f examples/guestbook/redis-master-service.yaml app=redis,role=master,tier=backend
  # prove role=master
  kube::test::get_object_assert 'services redis-master' "{{range$service_selector_field}}{{.}}:{{end}}" "redis:master:backend:"
  # Show dry-run works on running selector
  kubectl set selector services redis-master role=padawan --dry-run -o yaml "${kube_flags[@]}"
  ! kubectl set selector services redis-master role=padawan --local -o yaml "${kube_flags[@]}"
  kube::test::get_object_assert 'services redis-master' "{{range$service_selector_field}}{{.}}:{{end}}" "redis:master:backend:"

  ### Dump current redis-master service
  output_service=$(kubectl get service redis-master -o json --output-version=v1 "${kube_flags[@]}")

  ### Delete redis-master-service by id
  # Pre-condition: redis-master service exists
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:redis-master:'
  # Command
  kubectl delete service redis-master "${kube_flags[@]}"
  if [[ "${WAIT_FOR_DELETION:-}" == "true" ]]; then
    kube::test::wait_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:'
  fi
  # Post-condition: Only the default kubernetes services exist
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:'

  ### Create redis-master-service from dumped JSON
  # Pre-condition: Only the default kubernetes services exist
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:'
  # Command
  echo "${output_service}" | kubectl create -f - "${kube_flags[@]}"
  # Post-condition: redis-master service is created
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:redis-master:'

  ### Create redis-master-v1-test service
  # Pre-condition: redis-master-service service exists
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:redis-master:'
  # Command
  kubectl create -f - "${kube_flags[@]}" << __EOF__
{
  "kind": "Service",
  "apiVersion": "v1",
  "metadata": {
    "name": "service-v1-test"
  },
  "spec": {
    "ports": [
      {
        "protocol": "TCP",
        "port": 80,
        "targetPort": 80
      }
    ]
  }
}
__EOF__
  # Post-condition: service-v1-test service is created
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:redis-master:service-.*-test:'

  ### Identity
  kubectl get service "${kube_flags[@]}" service-v1-test -o json | kubectl replace "${kube_flags[@]}" -f -

  ### Delete services by id
  # Pre-condition: service-v1-test exists
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:redis-master:service-.*-test:'
  # Command
  kubectl delete service redis-master "${kube_flags[@]}"
  kubectl delete service "service-v1-test" "${kube_flags[@]}"
  if [[ "${WAIT_FOR_DELETION:-}" == "true" ]]; then
    kube::test::wait_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:'
  fi
  # Post-condition: Only the default kubernetes services exist
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:'

  ### Create two services
  # Pre-condition: Only the default kubernetes services exist
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:'
  # Command
  kubectl create -f examples/guestbook/redis-master-service.yaml "${kube_flags[@]}"
  kubectl create -f examples/guestbook/redis-slave-service.yaml "${kube_flags[@]}"
  # Post-condition: redis-master and redis-slave services are created
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:redis-master:redis-slave:'

  ### Custom columns can be specified
  # Pre-condition: generate output using custom columns
  output_message=$(kubectl get services -o=custom-columns=NAME:.metadata.name,RSRC:.metadata.resourceVersion 2>&1 "${kube_flags[@]}")
  # Post-condition: should contain name column
  kube::test::if_has_string "${output_message}" 'redis-master'

  ### Delete multiple services at once
  # Pre-condition: redis-master and redis-slave services exist
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:redis-master:redis-slave:'
  # Command
  kubectl delete services redis-master redis-slave "${kube_flags[@]}" # delete multiple services at once
  if [[ "${WAIT_FOR_DELETION:-}" == "true" ]]; then
    kube::test::wait_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:'
  fi
  # Post-condition: Only the default kubernetes services exist
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:'

  ### Create an ExternalName service
  # Pre-condition: Only the default kubernetes service exist
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:'
  # Command
  kubectl create service externalname beep-boop --external-name bar.com
  # Post-condition: beep-boop service is created
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'beep-boop:kubernetes:'

  ### Delete beep-boop service by id
  # Pre-condition: beep-boop service exists
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'beep-boop:kubernetes:'
  # Command
  kubectl delete service beep-boop "${kube_flags[@]}"
  if [[ "${WAIT_FOR_DELETION:-}" == "true" ]]; then
    kube::test::wait_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:'
  fi
  # Post-condition: Only the default kubernetes services exist
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:'

  set +o nounset
  set +o errexit
}

run_rc_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl(v1:replicationcontrollers)"

  ### Create and stop controller, make sure it doesn't leak pods
  # Pre-condition: no replication controller exists
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f hack/testdata/frontend-controller.yaml "${kube_flags[@]}"
  kubectl delete rc frontend "${kube_flags[@]}"
  # Post-condition: no pods from frontend controller
  kube::test::get_object_assert 'pods -l "name=frontend"' "{{range.items}}{{$id_field}}:{{end}}" ''

  ### Create replication controller frontend from JSON
  # Pre-condition: no replication controller exists
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f hack/testdata/frontend-controller.yaml "${kube_flags[@]}"
  # Post-condition: frontend replication controller is created
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" 'frontend:'
  # Describe command should print detailed information
  kube::test::describe_object_assert rc 'frontend' "Name:" "Pod Template:" "Labels:" "Selector:" "Replicas:" "Pods Status:" "Volumes:" "GET_HOSTS_FROM:"
  # Describe command should print events information by default
  kube::test::describe_object_events_assert rc 'frontend'
  # Describe command should not print events information when show-events=false
  kube::test::describe_object_events_assert rc 'frontend' false
  # Describe command should print events information when show-events=true
  kube::test::describe_object_events_assert rc 'frontend' true
  # Describe command (resource only) should print detailed information
  kube::test::describe_resource_assert rc "Name:" "Name:" "Pod Template:" "Labels:" "Selector:" "Replicas:" "Pods Status:" "Volumes:" "GET_HOSTS_FROM:"
  # Describe command should print events information by default
  kube::test::describe_resource_events_assert rc
  # Describe command should not print events information when show-events=false
  kube::test::describe_resource_events_assert rc false
  # Describe command should print events information when show-events=true
  kube::test::describe_resource_events_assert rc true

  ### Scale replication controller frontend with current-replicas and replicas
  # Pre-condition: 3 replicas
  kube::test::get_object_assert 'rc frontend' "{{$rc_replicas_field}}" '3'
  # Command
  kubectl scale --current-replicas=3 --replicas=2 replicationcontrollers frontend "${kube_flags[@]}"
  # Post-condition: 2 replicas
  kube::test::get_object_assert 'rc frontend' "{{$rc_replicas_field}}" '2'

  ### Scale replication controller frontend with (wrong) current-replicas and replicas
  # Pre-condition: 2 replicas
  kube::test::get_object_assert 'rc frontend' "{{$rc_replicas_field}}" '2'
  # Command
  ! kubectl scale --current-replicas=3 --replicas=2 replicationcontrollers frontend "${kube_flags[@]}"
  # Post-condition: nothing changed
  kube::test::get_object_assert 'rc frontend' "{{$rc_replicas_field}}" '2'

  ### Scale replication controller frontend with replicas only
  # Pre-condition: 2 replicas
  kube::test::get_object_assert 'rc frontend' "{{$rc_replicas_field}}" '2'
  # Command
  kubectl scale  --replicas=3 replicationcontrollers frontend "${kube_flags[@]}"
  # Post-condition: 3 replicas
  kube::test::get_object_assert 'rc frontend' "{{$rc_replicas_field}}" '3'

  ### Scale replication controller from JSON with replicas only
  # Pre-condition: 3 replicas
  kube::test::get_object_assert 'rc frontend' "{{$rc_replicas_field}}" '3'
  # Command
  kubectl scale  --replicas=2 -f hack/testdata/frontend-controller.yaml "${kube_flags[@]}"
  # Post-condition: 2 replicas
  kube::test::get_object_assert 'rc frontend' "{{$rc_replicas_field}}" '2'
  # Clean-up
  kubectl delete rc frontend "${kube_flags[@]}"

  ### Scale multiple replication controllers
  kubectl create -f examples/guestbook/legacy/redis-master-controller.yaml "${kube_flags[@]}"
  kubectl create -f examples/guestbook/legacy/redis-slave-controller.yaml "${kube_flags[@]}"
  # Command
  kubectl scale rc/redis-master rc/redis-slave --replicas=4 "${kube_flags[@]}"
  # Post-condition: 4 replicas each
  kube::test::get_object_assert 'rc redis-master' "{{$rc_replicas_field}}" '4'
  kube::test::get_object_assert 'rc redis-slave' "{{$rc_replicas_field}}" '4'
  # Clean-up
  kubectl delete rc redis-{master,slave} "${kube_flags[@]}"

  ### Scale a job
  kubectl create -f test/fixtures/doc-yaml/user-guide/job.yaml "${kube_flags[@]}"
  # Command
  kubectl scale --replicas=2 job/pi
  # Post-condition: 2 replicas for pi
  kube::test::get_object_assert 'job pi' "{{$job_parallelism_field}}" '2'
  # Clean-up
  kubectl delete job/pi "${kube_flags[@]}"

  ### Scale a deployment
  kubectl create -f test/fixtures/doc-yaml/user-guide/deployment.yaml "${kube_flags[@]}"
  # Command
  kubectl scale --current-replicas=3 --replicas=1 deployment/nginx-deployment
  # Post-condition: 1 replica for nginx-deployment
  kube::test::get_object_assert 'deployment nginx-deployment' "{{$deployment_replicas}}" '1'
  # Clean-up
  kubectl delete deployment/nginx-deployment "${kube_flags[@]}"

  ### Expose a deployment as a service
  kubectl create -f test/fixtures/doc-yaml/user-guide/deployment.yaml "${kube_flags[@]}"
  # Pre-condition: 3 replicas
  kube::test::get_object_assert 'deployment nginx-deployment' "{{$deployment_replicas}}" '3'
  # Command
  kubectl expose deployment/nginx-deployment
  # Post-condition: service exists and exposes deployment port (80)
  kube::test::get_object_assert 'service nginx-deployment' "{{$port_field}}" '80'
  # Clean-up
  kubectl delete deployment/nginx-deployment service/nginx-deployment "${kube_flags[@]}"

  ### Expose replication controller as service
  kubectl create -f hack/testdata/frontend-controller.yaml "${kube_flags[@]}"
  # Pre-condition: 3 replicas
  kube::test::get_object_assert 'rc frontend' "{{$rc_replicas_field}}" '3'
  # Command
  kubectl expose rc frontend --port=80 "${kube_flags[@]}"
  # Post-condition: service exists and the port is unnamed
  kube::test::get_object_assert 'service frontend' "{{$port_name}} {{$port_field}}" '<no value> 80'
  # Command
  kubectl expose service frontend --port=443 --name=frontend-2 "${kube_flags[@]}"
  # Post-condition: service exists and the port is unnamed
  kube::test::get_object_assert 'service frontend-2' "{{$port_name}} {{$port_field}}" '<no value> 443'
  # Command
  kubectl create -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml "${kube_flags[@]}"
  kubectl expose pod valid-pod --port=444 --name=frontend-3 "${kube_flags[@]}"
  # Post-condition: service exists and the port is unnamed
  kube::test::get_object_assert 'service frontend-3' "{{$port_name}} {{$port_field}}" '<no value> 444'
  # Create a service using service/v1 generator
  kubectl expose rc frontend --port=80 --name=frontend-4 --generator=service/v1 "${kube_flags[@]}"
  # Post-condition: service exists and the port is named default.
  kube::test::get_object_assert 'service frontend-4' "{{$port_name}} {{$port_field}}" 'default 80'
  # Verify that expose service works without specifying a port.
  kubectl expose service frontend --name=frontend-5 "${kube_flags[@]}"
  # Post-condition: service exists with the same port as the original service.
  kube::test::get_object_assert 'service frontend-5' "{{$port_field}}" '80'
  # Cleanup services
  kubectl delete pod valid-pod "${kube_flags[@]}"
  kubectl delete service frontend{,-2,-3,-4,-5} "${kube_flags[@]}"

  ### Expose negative invalid resource test
  # Pre-condition: don't need
  # Command
  output_message=$(! kubectl expose nodes 127.0.0.1 2>&1 "${kube_flags[@]}")
  # Post-condition: the error message has "cannot expose" string
  kube::test::if_has_string "${output_message}" 'cannot expose'

  ### Try to generate a service with invalid name (exceeding maximum valid size)
  # Pre-condition: use --name flag
  output_message=$(! kubectl expose -f hack/testdata/pod-with-large-name.yaml --name=invalid-large-service-name-that-has-more-than-sixty-three-characters --port=8081 2>&1 "${kube_flags[@]}")
  # Post-condition: should fail due to invalid name
  kube::test::if_has_string "${output_message}" 'metadata.name: Invalid value'
  # Pre-condition: default run without --name flag; should succeed by truncating the inherited name
  output_message=$(kubectl expose -f hack/testdata/pod-with-large-name.yaml --port=8081 2>&1 "${kube_flags[@]}")
  # Post-condition: inherited name from pod has been truncated
  kube::test::if_has_string "${output_message}" '\"kubernetes-serve-hostname-testing-sixty-three-characters-in-len\" exposed'
  # Clean-up
  kubectl delete svc kubernetes-serve-hostname-testing-sixty-three-characters-in-len "${kube_flags[@]}"

  ### Expose multiport object as a new service
  # Pre-condition: don't use --port flag
  output_message=$(kubectl expose -f test/fixtures/doc-yaml/admin/high-availability/etcd.yaml --selector=test=etcd 2>&1 "${kube_flags[@]}")
  # Post-condition: expose succeeded
  kube::test::if_has_string "${output_message}" '\"etcd-server\" exposed'
  # Post-condition: generated service has both ports from the exposed pod
  kube::test::get_object_assert 'service etcd-server' "{{$port_name}} {{$port_field}}" 'port-1 2380'
  kube::test::get_object_assert 'service etcd-server' "{{$second_port_name}} {{$second_port_field}}" 'port-2 2379'
  # Clean-up
  kubectl delete svc etcd-server "${kube_flags[@]}"

  ### Delete replication controller with id
  # Pre-condition: frontend replication controller exists
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" 'frontend:'
  # Command
  kubectl delete rc frontend "${kube_flags[@]}"
  # Post-condition: no replication controller exists
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" ''

  ### Create two replication controllers
  # Pre-condition: no replication controller exists
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f hack/testdata/frontend-controller.yaml "${kube_flags[@]}"
  kubectl create -f examples/guestbook/legacy/redis-slave-controller.yaml "${kube_flags[@]}"
  # Post-condition: frontend and redis-slave
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" 'frontend:redis-slave:'

  ### Delete multiple controllers at once
  # Pre-condition: frontend and redis-slave
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" 'frontend:redis-slave:'
  # Command
  kubectl delete rc frontend redis-slave "${kube_flags[@]}" # delete multiple controllers at once
  # Post-condition: no replication controller exists
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" ''

  ### Auto scale replication controller
  # Pre-condition: no replication controller exists
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f hack/testdata/frontend-controller.yaml "${kube_flags[@]}"
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" 'frontend:'
  # autoscale 1~2 pods, CPU utilization 70%, rc specified by file
  kubectl autoscale -f hack/testdata/frontend-controller.yaml "${kube_flags[@]}" --max=2 --cpu-percent=70
  kube::test::get_object_assert 'hpa frontend' "{{$hpa_min_field}} {{$hpa_max_field}} {{$hpa_cpu_field}}" '1 2 70'
  kubectl delete hpa frontend "${kube_flags[@]}"
  # autoscale 2~3 pods, no CPU utilization specified, rc specified by name
  kubectl autoscale rc frontend "${kube_flags[@]}" --min=2 --max=3
  kube::test::get_object_assert 'hpa frontend' "{{$hpa_min_field}} {{$hpa_max_field}} {{$hpa_cpu_field}}" '2 3 80'
  kubectl delete hpa frontend "${kube_flags[@]}"
  # autoscale without specifying --max should fail
  ! kubectl autoscale rc frontend "${kube_flags[@]}"
  # Clean up
  kubectl delete rc frontend "${kube_flags[@]}"

  ## Set resource limits/request of a deployment
  # Pre-condition: no deployment exists
  kube::test::get_object_assert deployment "{{range.items}}{{$id_field}}:{{end}}" ''
  # Set resources of a local file without talking to the server
  kubectl set resources -f hack/testdata/deployment-multicontainer-resources.yaml -c=perl --limits=cpu=300m --requests=cpu=300m --local -o yaml "${kube_flags[@]}"
  ! kubectl set resources -f hack/testdata/deployment-multicontainer-resources.yaml -c=perl --limits=cpu=300m --requests=cpu=300m --dry-run -o yaml "${kube_flags[@]}"
  # Create a deployment
  kubectl create -f hack/testdata/deployment-multicontainer-resources.yaml "${kube_flags[@]}"
  kube::test::get_object_assert deployment "{{range.items}}{{$id_field}}:{{end}}" 'nginx-deployment-resources:'
  kube::test::get_object_assert deployment "{{range.items}}{{$image_field0}}:{{end}}" "${IMAGE_DEPLOYMENT_R1}:"
  kube::test::get_object_assert deployment "{{range.items}}{{$image_field1}}:{{end}}" "${IMAGE_PERL}:"
  # Set the deployment's cpu limits
  kubectl set resources deployment nginx-deployment-resources --limits=cpu=100m "${kube_flags[@]}"
  kube::test::get_object_assert deployment "{{range.items}}{{(index .spec.template.spec.containers 0).resources.limits.cpu}}:{{end}}" "100m:"
  kube::test::get_object_assert deployment "{{range.items}}{{(index .spec.template.spec.containers 1).resources.limits.cpu}}:{{end}}" "100m:"
  # Set a non-existing container should fail
  ! kubectl set resources deployment nginx-deployment-resources -c=redis --limits=cpu=100m
  # Set the limit of a specific container in deployment
  kubectl set resources deployment nginx-deployment-resources -c=nginx --limits=cpu=200m "${kube_flags[@]}"
  kube::test::get_object_assert deployment "{{range.items}}{{(index .spec.template.spec.containers 0).resources.limits.cpu}}:{{end}}" "200m:"
  kube::test::get_object_assert deployment "{{range.items}}{{(index .spec.template.spec.containers 1).resources.limits.cpu}}:{{end}}" "100m:"
  # Set limits/requests of a deployment specified by a file
  kubectl set resources -f hack/testdata/deployment-multicontainer-resources.yaml -c=perl --limits=cpu=300m --requests=cpu=300m "${kube_flags[@]}"
  kube::test::get_object_assert deployment "{{range.items}}{{(index .spec.template.spec.containers 0).resources.limits.cpu}}:{{end}}" "200m:"
  kube::test::get_object_assert deployment "{{range.items}}{{(index .spec.template.spec.containers 1).resources.limits.cpu}}:{{end}}" "300m:"
  kube::test::get_object_assert deployment "{{range.items}}{{(index .spec.template.spec.containers 1).resources.requests.cpu}}:{{end}}" "300m:"
  # Show dry-run works on running deployments
  kubectl set resources deployment nginx-deployment-resources -c=perl --limits=cpu=400m --requests=cpu=400m --dry-run -o yaml "${kube_flags[@]}"
  ! kubectl set resources deployment nginx-deployment-resources -c=perl --limits=cpu=400m --requests=cpu=400m --local -o yaml "${kube_flags[@]}"
  kube::test::get_object_assert deployment "{{range.items}}{{(index .spec.template.spec.containers 0).resources.limits.cpu}}:{{end}}" "200m:"
  kube::test::get_object_assert deployment "{{range.items}}{{(index .spec.template.spec.containers 1).resources.limits.cpu}}:{{end}}" "300m:"
  kube::test::get_object_assert deployment "{{range.items}}{{(index .spec.template.spec.containers 1).resources.requests.cpu}}:{{end}}" "300m:"
  # Clean up
  kubectl delete deployment nginx-deployment-resources "${kube_flags[@]}"

  set +o nounset
  set +o errexit
}

run_deployment_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing deployments"
  # Test kubectl create deployment (using default - old generator)
  kubectl create deployment test-nginx-extensions --image=gcr.io/google-containers/nginx:test-cmd
  # Post-Condition: Deployment "nginx" is created.
  kube::test::get_object_assert 'deploy test-nginx-extensions' "{{$container_name_field}}" 'nginx'
  # and old generator was used, iow. old defaults are applied
  output_message=$(kubectl get deployment.extensions/test-nginx-extensions -o jsonpath='{.spec.revisionHistoryLimit}')
  kube::test::if_has_not_string "${output_message}" '2'
  # Ensure we can interact with deployments through extensions and apps endpoints
  output_message=$(kubectl get deployment.extensions -o=jsonpath='{.items[0].apiVersion}' 2>&1 "${kube_flags[@]}")
  kube::test::if_has_string "${output_message}" 'extensions/v1beta1'
  output_message=$(kubectl get deployment.apps -o=jsonpath='{.items[0].apiVersion}' 2>&1 "${kube_flags[@]}")
  kube::test::if_has_string "${output_message}" 'apps/v1beta1'
  # Clean up
  kubectl delete deployment test-nginx-extensions "${kube_flags[@]}"

  # Test kubectl create deployment
  kubectl create deployment test-nginx-apps --image=gcr.io/google-containers/nginx:test-cmd --generator=deployment-basic/apps.v1beta1
  # Post-Condition: Deployment "nginx" is created.
  kube::test::get_object_assert 'deploy test-nginx-apps' "{{$container_name_field}}" 'nginx'
  # and new generator was used, iow. new defaults are applied
  output_message=$(kubectl get deployment/test-nginx-apps -o jsonpath='{.spec.revisionHistoryLimit}')
  kube::test::if_has_string "${output_message}" '2'
  # Ensure we can interact with deployments through extensions and apps endpoints
  output_message=$(kubectl get deployment.extensions -o=jsonpath='{.items[0].apiVersion}' 2>&1 "${kube_flags[@]}")
  kube::test::if_has_string "${output_message}" 'extensions/v1beta1'
  output_message=$(kubectl get deployment.apps -o=jsonpath='{.items[0].apiVersion}' 2>&1 "${kube_flags[@]}")
  kube::test::if_has_string "${output_message}" 'apps/v1beta1'
  # Describe command (resource only) should print detailed information
  kube::test::describe_resource_assert rs "Name:" "Pod Template:" "Labels:" "Selector:" "Controlled By" "Replicas:" "Pods Status:" "Volumes:"
  # Describe command (resource only) should print detailed information
  kube::test::describe_resource_assert pods "Name:" "Image:" "Node:" "Labels:" "Status:" "Created By" "Controlled By"
  # Clean up
  kubectl delete deployment test-nginx-apps "${kube_flags[@]}"

  ### Test kubectl create deployment should not fail validation
  # Pre-Condition: No deployment exists.
  kube::test::get_object_assert deployment "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f hack/testdata/deployment-with-UnixUserID.yaml "${kube_flags[@]}"
  # Post-Condition: Deployment "deployment-with-unixuserid" is created.
  kube::test::get_object_assert deployment "{{range.items}}{{$id_field}}:{{end}}" 'deployment-with-unixuserid:'
  # Clean up
  kubectl delete deployment deployment-with-unixuserid "${kube_flags[@]}"

  ### Test cascading deletion
  ## Test that rs is deleted when deployment is deleted.
  # Pre-condition: no deployment exists
  kube::test::get_object_assert deployment "{{range.items}}{{$id_field}}:{{end}}" ''
  # Create deployment
  kubectl create -f test/fixtures/doc-yaml/user-guide/deployment.yaml "${kube_flags[@]}"
  # Wait for rs to come up.
  kube::test::wait_object_assert rs "{{range.items}}{{$rs_replicas_field}}{{end}}" '3'
  # Deleting the deployment should delete the rs.
  kubectl delete deployment nginx-deployment "${kube_flags[@]}"
  kube::test::wait_object_assert rs "{{range.items}}{{$id_field}}:{{end}}" ''

  ## Test that rs is not deleted when deployment is deleted with cascade set to false.
  # Pre-condition: no deployment and rs exist
  kube::test::get_object_assert deployment "{{range.items}}{{$id_field}}:{{end}}" ''
  kube::test::get_object_assert rs "{{range.items}}{{$id_field}}:{{end}}" ''
  # Create deployment
  kubectl create deployment nginx-deployment --image=gcr.io/google-containers/nginx:test-cmd
  # Wait for rs to come up.
  kube::test::wait_object_assert rs "{{range.items}}{{$rs_replicas_field}}{{end}}" '1'
  # Delete the deployment with cascade set to false.
  kubectl delete deployment nginx-deployment "${kube_flags[@]}" --cascade=false
  # Wait for the deployment to be deleted and then verify that rs is not
  # deleted.
  kube::test::wait_object_assert deployment "{{range.items}}{{$id_field}}:{{end}}" ''
  kube::test::get_object_assert rs "{{range.items}}{{$rs_replicas_field}}{{end}}" '1'
  # Cleanup
  # Find the name of the rs to be deleted.
  output_message=$(kubectl get rs "${kube_flags[@]}" -o template --template={{range.items}}{{$id_field}}{{end}})
  kubectl delete rs ${output_message} "${kube_flags[@]}"

  ### Auto scale deployment
  # Pre-condition: no deployment exists
  kube::test::get_object_assert deployment "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f test/fixtures/doc-yaml/user-guide/deployment.yaml "${kube_flags[@]}"
  kube::test::get_object_assert deployment "{{range.items}}{{$id_field}}:{{end}}" 'nginx-deployment:'
  # autoscale 2~3 pods, no CPU utilization specified
  kubectl-with-retry autoscale deployment nginx-deployment "${kube_flags[@]}" --min=2 --max=3
  kube::test::get_object_assert 'hpa nginx-deployment' "{{$hpa_min_field}} {{$hpa_max_field}} {{$hpa_cpu_field}}" '2 3 80'
  # Clean up
  # Note that we should delete hpa first, otherwise it may fight with the deployment reaper.
  kubectl delete hpa nginx-deployment "${kube_flags[@]}"
  kubectl delete deployment.extensions nginx-deployment "${kube_flags[@]}"

  ### Rollback a deployment
  # Pre-condition: no deployment exists
  kube::test::get_object_assert deployment "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  # Create a deployment (revision 1)
  kubectl create -f hack/testdata/deployment-revision1.yaml "${kube_flags[@]}"
  kube::test::get_object_assert deployment "{{range.items}}{{$id_field}}:{{end}}" 'nginx:'
  kube::test::get_object_assert deployment "{{range.items}}{{$image_field0}}:{{end}}" "${IMAGE_DEPLOYMENT_R1}:"
  # Rollback to revision 1 - should be no-op
  kubectl rollout undo deployment nginx --to-revision=1 "${kube_flags[@]}"
  kube::test::get_object_assert deployment "{{range.items}}{{$image_field0}}:{{end}}" "${IMAGE_DEPLOYMENT_R1}:"
  # Update the deployment (revision 2)
  kubectl apply -f hack/testdata/deployment-revision2.yaml "${kube_flags[@]}"
  kube::test::get_object_assert deployment.extensions "{{range.items}}{{$image_field0}}:{{end}}" "${IMAGE_DEPLOYMENT_R2}:"
  # Rollback to revision 1 with dry-run - should be no-op
  kubectl rollout undo deployment nginx --dry-run=true "${kube_flags[@]}" | grep "test-cmd"
  kube::test::get_object_assert deployment.extensions "{{range.items}}{{$image_field0}}:{{end}}" "${IMAGE_DEPLOYMENT_R2}:"
  # Rollback to revision 1
  kubectl rollout undo deployment nginx --to-revision=1 "${kube_flags[@]}"
  sleep 1
  kube::test::get_object_assert deployment "{{range.items}}{{$image_field0}}:{{end}}" "${IMAGE_DEPLOYMENT_R1}:"
  # Rollback to revision 1000000 - should be no-op
  kubectl rollout undo deployment nginx --to-revision=1000000 "${kube_flags[@]}"
  kube::test::get_object_assert deployment "{{range.items}}{{$image_field0}}:{{end}}" "${IMAGE_DEPLOYMENT_R1}:"
  # Rollback to last revision
  kubectl rollout undo deployment nginx "${kube_flags[@]}"
  sleep 1
  kube::test::get_object_assert deployment "{{range.items}}{{$image_field0}}:{{end}}" "${IMAGE_DEPLOYMENT_R2}:"
  # Pause the deployment
  kubectl-with-retry rollout pause deployment nginx "${kube_flags[@]}"
  # A paused deployment cannot be rolled back
  ! kubectl rollout undo deployment nginx "${kube_flags[@]}"
  # Resume the deployment
  kubectl-with-retry rollout resume deployment nginx "${kube_flags[@]}"
  # The resumed deployment can now be rolled back
  kubectl rollout undo deployment nginx "${kube_flags[@]}"
  # Check that the new replica set has all old revisions stored in an annotation
  newrs="$(kubectl describe deployment nginx | grep NewReplicaSet | awk '{print $2}')"
  kubectl get rs "${newrs}" -o yaml | grep "deployment.kubernetes.io/revision-history: 1,3"
  # Check that trying to watch the status of a superseded revision returns an error
  ! kubectl rollout status deployment/nginx --revision=3
  cat hack/testdata/deployment-revision1.yaml | $SED "s/name: nginx$/name: nginx2/" | kubectl create -f - "${kube_flags[@]}"
  # Deletion of both deployments should not be blocked
  kubectl delete deployment nginx2 "${kube_flags[@]}"
  # Clean up
  kubectl delete deployment nginx "${kube_flags[@]}"

  ### Set image of a deployment
  # Pre-condition: no deployment exists
  kube::test::get_object_assert deployment "{{range.items}}{{$id_field}}:{{end}}" ''
  # Create a deployment
  kubectl create -f hack/testdata/deployment-multicontainer.yaml "${kube_flags[@]}"
  kube::test::get_object_assert deployment "{{range.items}}{{$id_field}}:{{end}}" 'nginx-deployment:'
  kube::test::get_object_assert deployment "{{range.items}}{{$image_field0}}:{{end}}" "${IMAGE_DEPLOYMENT_R1}:"
  kube::test::get_object_assert deployment "{{range.items}}{{$image_field1}}:{{end}}" "${IMAGE_PERL}:"
  # Set the deployment's image
  kubectl set image deployment nginx-deployment nginx="${IMAGE_DEPLOYMENT_R2}" "${kube_flags[@]}"
  kube::test::get_object_assert deployment "{{range.items}}{{$image_field0}}:{{end}}" "${IMAGE_DEPLOYMENT_R2}:"
  kube::test::get_object_assert deployment "{{range.items}}{{$image_field1}}:{{end}}" "${IMAGE_PERL}:"
  # Set non-existing container should fail
  ! kubectl set image deployment nginx-deployment redis=redis "${kube_flags[@]}"
  # Set image of deployments without specifying name
  kubectl set image deployments --all nginx="${IMAGE_DEPLOYMENT_R1}" "${kube_flags[@]}"
  kube::test::get_object_assert deployment "{{range.items}}{{$image_field0}}:{{end}}" "${IMAGE_DEPLOYMENT_R1}:"
  kube::test::get_object_assert deployment "{{range.items}}{{$image_field1}}:{{end}}" "${IMAGE_PERL}:"
  # Set image of a deployment specified by file
  kubectl set image -f hack/testdata/deployment-multicontainer.yaml nginx="${IMAGE_DEPLOYMENT_R2}" "${kube_flags[@]}"
  kube::test::get_object_assert deployment "{{range.items}}{{$image_field0}}:{{end}}" "${IMAGE_DEPLOYMENT_R2}:"
  kube::test::get_object_assert deployment "{{range.items}}{{$image_field1}}:{{end}}" "${IMAGE_PERL}:"
  # Set image of a local file without talking to the server
  kubectl set image -f hack/testdata/deployment-multicontainer.yaml nginx="${IMAGE_DEPLOYMENT_R1}" "${kube_flags[@]}" --local -o yaml
  kube::test::get_object_assert deployment "{{range.items}}{{$image_field0}}:{{end}}" "${IMAGE_DEPLOYMENT_R2}:"
  kube::test::get_object_assert deployment "{{range.items}}{{$image_field1}}:{{end}}" "${IMAGE_PERL}:"
  # Set image of all containers of the deployment
  kubectl set image deployment nginx-deployment "*"="${IMAGE_DEPLOYMENT_R1}" "${kube_flags[@]}"
  kube::test::get_object_assert deployment "{{range.items}}{{$image_field0}}:{{end}}" "${IMAGE_DEPLOYMENT_R1}:"
  kube::test::get_object_assert deployment "{{range.items}}{{$image_field1}}:{{end}}" "${IMAGE_DEPLOYMENT_R1}:"
  # Set image of all containners of the deployment again when image not change
  kubectl set image deployment nginx-deployment "*"="${IMAGE_DEPLOYMENT_R1}" "${kube_flags[@]}"
  kube::test::get_object_assert deployment "{{range.items}}{{$image_field0}}:{{end}}" "${IMAGE_DEPLOYMENT_R1}:"
  kube::test::get_object_assert deployment "{{range.items}}{{$image_field1}}:{{end}}" "${IMAGE_DEPLOYMENT_R1}:"
  # Clean up
  kubectl delete deployment nginx-deployment "${kube_flags[@]}"

  ### Set env of a deployment
  # Pre-condition: no deployment exists
  kube::test::get_object_assert deployment "{{range.items}}{{$id_field}}:{{end}}" ''
  # Create a deployment
  kubectl create -f hack/testdata/deployment-multicontainer.yaml "${kube_flags[@]}"
  kubectl create -f hack/testdata/configmap.yaml "${kube_flags[@]}"
  kubectl create -f hack/testdata/secret.yaml "${kube_flags[@]}"
  kube::test::get_object_assert deployment "{{range.items}}{{$id_field}}:{{end}}" 'nginx-deployment:'
  kube::test::get_object_assert configmap "{{range.items}}{{$id_field}}:{{end}}" 'test-set-env-config:'
  kube::test::get_object_assert secret "{{range.items}}{{$id_field}}:{{end}}" 'test-set-env-secret:'
  # Set env of deployments for all container
  kubectl set env deployment nginx-deployment env=prod "${kube_flags[@]}"
  # Set env of deployments for specific container
  kubectl set env deployment nginx-deployment env=prod -c=nginx "${kube_flags[@]}"
  # Set env of deployments by configmap
  kubectl set env deployment nginx-deployment --from=configmap/test-set-env-config "${kube_flags[@]}"
  # Set env of deployments by secret
  kubectl set env deployment nginx-deployment --from=secret/test-set-env-secret "${kube_flags[@]}"
  # Remove specific env of deployment
  kubectl set env deployment nginx-deployment env-
  # Clean up
  kubectl delete deployment nginx-deployment "${kube_flags[@]}"
  kubectl delete configmap test-set-env-config "${kube_flags[@]}"
  kubectl delete secret test-set-env-secret "${kube_flags[@]}"

  ### Delete a deployment with initializer
  # Pre-condition: no deployment exists
  kube::test::get_object_assert deployment "{{range.items}}{{$id_field}}:{{end}}" ''
  # Create a deployment
  kubectl create --request-timeout=1 -f hack/testdata/deployment-with-initializer.yaml 2>&1 "${kube_flags[@]}" || true
  kube::test::get_object_assert 'deployment web' "{{$id_field}}" 'web'
  # Delete a deployment
  kubectl delete deployment web "${kube_flags[@]}"
  # Check Deployment web doesn't exist
  output_message=$(! kubectl get deployment web 2>&1 "${kube_flags[@]}")
  kube::test::if_has_string "${output_message}" '"web" not found'

  set +o nounset
  set +o errexit
}

run_rs_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl(v1:replicasets)"

  ### Create and stop a replica set, make sure it doesn't leak pods
  # Pre-condition: no replica set exists
  kube::test::get_object_assert rs "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f hack/testdata/frontend-replicaset.yaml "${kube_flags[@]}"
  kube::log::status "Deleting rs"
  kubectl delete rs frontend "${kube_flags[@]}"
  # Post-condition: no pods from frontend replica set
  kube::test::get_object_assert 'pods -l "tier=frontend"' "{{range.items}}{{$id_field}}:{{end}}" ''

  ### Create and then delete a replica set with cascade=false, make sure it doesn't delete pods.
  # Pre-condition: no replica set exists
  kube::test::get_object_assert rs "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f hack/testdata/frontend-replicaset.yaml "${kube_flags[@]}"
  kube::log::status "Deleting rs"
  kubectl delete rs frontend "${kube_flags[@]}" --cascade=false
  # Wait for the rs to be deleted.
  kube::test::wait_object_assert rs "{{range.items}}{{$id_field}}:{{end}}" ''
  # Post-condition: All 3 pods still remain from frontend replica set
  kube::test::get_object_assert 'pods -l "tier=frontend"' "{{range.items}}{{$pod_container_name_field}}:{{end}}" 'php-redis:php-redis:php-redis:'
  # Cleanup
  kubectl delete pods -l "tier=frontend" "${kube_flags[@]}"
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''

  ### Create replica set frontend from YAML
  # Pre-condition: no replica set exists
  kube::test::get_object_assert rs "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f hack/testdata/frontend-replicaset.yaml "${kube_flags[@]}"
  # Post-condition: frontend replica set is created
  kube::test::get_object_assert rs "{{range.items}}{{$id_field}}:{{end}}" 'frontend:'
  # Describe command should print detailed information
  kube::test::describe_object_assert rs 'frontend' "Name:" "Pod Template:" "Labels:" "Selector:" "Replicas:" "Pods Status:" "Volumes:"
  # Describe command should print events information by default
  kube::test::describe_object_events_assert rs 'frontend'
  # Describe command should not print events information when show-events=false
  kube::test::describe_object_events_assert rs 'frontend' false
  # Describe command should print events information when show-events=true
  kube::test::describe_object_events_assert rs 'frontend' true
  # Describe command (resource only) should print detailed information
  kube::test::describe_resource_assert rs "Name:" "Pod Template:" "Labels:" "Selector:" "Replicas:" "Pods Status:" "Volumes:"
  # Describe command should print events information by default
  kube::test::describe_resource_events_assert rs
  # Describe command should not print events information when show-events=false
  kube::test::describe_resource_events_assert rs false
  # Describe command should print events information when show-events=true
  kube::test::describe_resource_events_assert rs true
  # Describe command (resource only) should print detailed information
  kube::test::describe_resource_assert pods "Name:" "Image:" "Node:" "Labels:" "Status:" "Created By" "Controlled By"

  ### Scale replica set frontend with current-replicas and replicas
  # Pre-condition: 3 replicas
  kube::test::get_object_assert 'rs frontend' "{{$rs_replicas_field}}" '3'
  # Command
  kubectl scale --current-replicas=3 --replicas=2 replicasets frontend "${kube_flags[@]}"
  # Post-condition: 2 replicas
  kube::test::get_object_assert 'rs frontend' "{{$rs_replicas_field}}" '2'
  # Clean-up
  kubectl delete rs frontend "${kube_flags[@]}"

  ### Expose replica set as service
  kubectl create -f hack/testdata/frontend-replicaset.yaml "${kube_flags[@]}"
  # Pre-condition: 3 replicas
  kube::test::get_object_assert 'rs frontend' "{{$rs_replicas_field}}" '3'
  # Command
  kubectl expose rs frontend --port=80 "${kube_flags[@]}"
  # Post-condition: service exists and the port is unnamed
  kube::test::get_object_assert 'service frontend' "{{$port_name}} {{$port_field}}" '<no value> 80'
  # Create a service using service/v1 generator
  kubectl expose rs frontend --port=80 --name=frontend-2 --generator=service/v1 "${kube_flags[@]}"
  # Post-condition: service exists and the port is named default.
  kube::test::get_object_assert 'service frontend-2' "{{$port_name}} {{$port_field}}" 'default 80'
  # Cleanup services
  kubectl delete service frontend{,-2} "${kube_flags[@]}"

  ### Delete replica set with id
  # Pre-condition: frontend replica set exists
  kube::test::get_object_assert rs "{{range.items}}{{$id_field}}:{{end}}" 'frontend:'
  # Command
  kubectl delete rs frontend "${kube_flags[@]}"
  # Post-condition: no replica set exists
  kube::test::get_object_assert rs "{{range.items}}{{$id_field}}:{{end}}" ''

  ### Create two replica sets
  # Pre-condition: no replica set exists
  kube::test::get_object_assert rs "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f hack/testdata/frontend-replicaset.yaml "${kube_flags[@]}"
  kubectl create -f hack/testdata/redis-slave-replicaset.yaml "${kube_flags[@]}"
  # Post-condition: frontend and redis-slave
  kube::test::get_object_assert rs "{{range.items}}{{$id_field}}:{{end}}" 'frontend:redis-slave:'

  ### Delete multiple replica sets at once
  # Pre-condition: frontend and redis-slave
  kube::test::get_object_assert rs "{{range.items}}{{$id_field}}:{{end}}" 'frontend:redis-slave:'
  # Command
  kubectl delete rs frontend redis-slave "${kube_flags[@]}" # delete multiple replica sets at once
  # Post-condition: no replica set exists
  kube::test::get_object_assert rs "{{range.items}}{{$id_field}}:{{end}}" ''

  ### Delete a rs with initializer
  # Pre-condition: no rs exists
  kube::test::get_object_assert rs "{{range.items}}{{$id_field}}:{{end}}" ''
  # Create a rs
  kubectl create --request-timeout=1 -f hack/testdata/replicaset-with-initializer.yaml 2>&1 "${kube_flags[@]}" || true
  kube::test::get_object_assert 'rs nginx' "{{$id_field}}" 'nginx'
  # Delete a rs
  kubectl delete rs nginx "${kube_flags[@]}"
  # check rs nginx doesn't exist
  output_message=$(! kubectl get rs nginx 2>&1 "${kube_flags[@]}")
  kube::test::if_has_string "${output_message}" '"nginx" not found'

  if kube::test::if_supports_resource "${horizontalpodautoscalers}" ; then
    ### Auto scale replica set
    # Pre-condition: no replica set exists
    kube::test::get_object_assert rs "{{range.items}}{{$id_field}}:{{end}}" ''
    # Command
    kubectl create -f hack/testdata/frontend-replicaset.yaml "${kube_flags[@]}"
    kube::test::get_object_assert rs "{{range.items}}{{$id_field}}:{{end}}" 'frontend:'
    # autoscale 1~2 pods, CPU utilization 70%, replica set specified by file
    kubectl autoscale -f hack/testdata/frontend-replicaset.yaml "${kube_flags[@]}" --max=2 --cpu-percent=70
    kube::test::get_object_assert 'hpa frontend' "{{$hpa_min_field}} {{$hpa_max_field}} {{$hpa_cpu_field}}" '1 2 70'
    kubectl delete hpa frontend "${kube_flags[@]}"
    # autoscale 2~3 pods, no CPU utilization specified, replica set specified by name
    kubectl autoscale rs frontend "${kube_flags[@]}" --min=2 --max=3
    kube::test::get_object_assert 'hpa frontend' "{{$hpa_min_field}} {{$hpa_max_field}} {{$hpa_cpu_field}}" '2 3 80'
    kubectl delete hpa frontend "${kube_flags[@]}"
    # autoscale without specifying --max should fail
    ! kubectl autoscale rs frontend "${kube_flags[@]}"
    # Clean up
    kubectl delete rs frontend "${kube_flags[@]}"
  fi

  set +o nounset
  set +o errexit
}

run_daemonset_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl(v1:daemonsets)"

  ### Create a rolling update DaemonSet
  # Pre-condition: no DaemonSet exists
  kube::test::get_object_assert daemonsets "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl apply -f hack/testdata/rollingupdate-daemonset.yaml "${kube_flags[@]}"
  # Template Generation should be 1
  kube::test::get_object_assert 'daemonsets bind' "{{${template_generation_field}}}" '1'
  kubectl apply -f hack/testdata/rollingupdate-daemonset.yaml "${kube_flags[@]}"
  # Template Generation should stay 1
  kube::test::get_object_assert 'daemonsets bind' "{{${template_generation_field}}}" '1'
  # Clean up
  kubectl delete -f hack/testdata/rollingupdate-daemonset.yaml "${kube_flags[@]}"

  set +o nounset
  set +o errexit
}

run_daemonset_history_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl(v1:daemonsets, v1:controllerrevisions)"

  ### Test rolling back a DaemonSet
  # Pre-condition: no DaemonSet or its pods exists
  kube::test::get_object_assert daemonsets "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  # Create a DaemonSet (revision 1)
  kubectl apply -f hack/testdata/rollingupdate-daemonset.yaml --record "${kube_flags[@]}"
  kube::test::wait_object_assert controllerrevisions "{{range.items}}{{$annotations_field}}:{{end}}" ".*rollingupdate-daemonset.yaml --record.*"
  # Rollback to revision 1 - should be no-op
  kubectl rollout undo daemonset --to-revision=1 "${kube_flags[@]}"
  kube::test::get_object_assert daemonset "{{range.items}}{{$image_field0}}:{{end}}" "${IMAGE_PAUSE_V2}:"
  kube::test::get_object_assert daemonset "{{range.items}}{{$container_len}}{{end}}" "1"
  # Update the DaemonSet (revision 2)
  kubectl apply -f hack/testdata/rollingupdate-daemonset-rv2.yaml --record "${kube_flags[@]}"
  kube::test::wait_object_assert daemonset "{{range.items}}{{$image_field0}}:{{end}}" "${IMAGE_DAEMONSET_R2}:"
  kube::test::wait_object_assert daemonset "{{range.items}}{{$image_field1}}:{{end}}" "${IMAGE_DAEMONSET_R2_2}:"
  kube::test::get_object_assert daemonset "{{range.items}}{{$container_len}}{{end}}" "2"
  kube::test::wait_object_assert controllerrevisions "{{range.items}}{{$annotations_field}}:{{end}}" ".*rollingupdate-daemonset-rv2.yaml --record.*"
  # Rollback to revision 1 with dry-run - should be no-op
  kubectl rollout undo daemonset --dry-run=true "${kube_flags[@]}"
  kube::test::get_object_assert daemonset "{{range.items}}{{$image_field0}}:{{end}}" "${IMAGE_DAEMONSET_R2}:"
  kube::test::get_object_assert daemonset "{{range.items}}{{$image_field1}}:{{end}}" "${IMAGE_DAEMONSET_R2_2}:"
  kube::test::get_object_assert daemonset "{{range.items}}{{$container_len}}{{end}}" "2"
  # Rollback to revision 1
  kubectl rollout undo daemonset --to-revision=1 "${kube_flags[@]}"
  kube::test::wait_object_assert daemonset "{{range.items}}{{$image_field0}}:{{end}}" "${IMAGE_PAUSE_V2}:"
  kube::test::get_object_assert daemonset "{{range.items}}{{$container_len}}{{end}}" "1"
  # Rollback to revision 1000000 - should fail
  output_message=$(! kubectl rollout undo daemonset --to-revision=1000000 "${kube_flags[@]}" 2>&1)
  kube::test::if_has_string "${output_message}" "unable to find specified revision"
  kube::test::get_object_assert daemonset "{{range.items}}{{$image_field0}}:{{end}}" "${IMAGE_PAUSE_V2}:"
  kube::test::get_object_assert daemonset "{{range.items}}{{$container_len}}{{end}}" "1"
  # Rollback to last revision
  kubectl rollout undo daemonset "${kube_flags[@]}"
  kube::test::wait_object_assert daemonset "{{range.items}}{{$image_field0}}:{{end}}" "${IMAGE_DAEMONSET_R2}:"
  kube::test::wait_object_assert daemonset "{{range.items}}{{$image_field1}}:{{end}}" "${IMAGE_DAEMONSET_R2_2}:"
  kube::test::get_object_assert daemonset "{{range.items}}{{$container_len}}{{end}}" "2"
  # Clean up
  kubectl delete -f hack/testdata/rollingupdate-daemonset.yaml "${kube_flags[@]}"

  set +o nounset
  set +o errexit
}

run_statefulset_history_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl(v1:statefulsets, v1:controllerrevisions)"

  ### Test rolling back a StatefulSet
  # Pre-condition: no statefulset or its pods exists
  kube::test::get_object_assert statefulset "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  # Create a StatefulSet (revision 1)
  kubectl apply -f hack/testdata/rollingupdate-statefulset.yaml --record "${kube_flags[@]}"
  kube::test::wait_object_assert controllerrevisions "{{range.items}}{{$annotations_field}}:{{end}}" ".*rollingupdate-statefulset.yaml --record.*"
  # Rollback to revision 1 - should be no-op
  kubectl rollout undo statefulset --to-revision=1 "${kube_flags[@]}"
  kube::test::get_object_assert statefulset "{{range.items}}{{$image_field0}}:{{end}}" "${IMAGE_STATEFULSET_R1}:"
  kube::test::get_object_assert statefulset "{{range.items}}{{$container_len}}{{end}}" "1"
  # Update the statefulset (revision 2)
  kubectl apply -f hack/testdata/rollingupdate-statefulset-rv2.yaml --record "${kube_flags[@]}"
  kube::test::wait_object_assert statefulset "{{range.items}}{{$image_field0}}:{{end}}" "${IMAGE_STATEFULSET_R2}:"
  kube::test::wait_object_assert statefulset "{{range.items}}{{$image_field1}}:{{end}}" "${IMAGE_PAUSE_V2}:"
  kube::test::get_object_assert statefulset "{{range.items}}{{$container_len}}{{end}}" "2"
  kube::test::wait_object_assert controllerrevisions "{{range.items}}{{$annotations_field}}:{{end}}" ".*rollingupdate-statefulset-rv2.yaml --record.*"
  # Rollback to revision 1 with dry-run - should be no-op
  kubectl rollout undo statefulset --dry-run=true "${kube_flags[@]}"
  kube::test::get_object_assert statefulset "{{range.items}}{{$image_field0}}:{{end}}" "${IMAGE_STATEFULSET_R2}:"
  kube::test::get_object_assert statefulset "{{range.items}}{{$image_field1}}:{{end}}" "${IMAGE_PAUSE_V2}:"
  kube::test::get_object_assert statefulset "{{range.items}}{{$container_len}}{{end}}" "2"
  # Rollback to revision 1
  kubectl rollout undo statefulset --to-revision=1 "${kube_flags[@]}"
  kube::test::wait_object_assert statefulset "{{range.items}}{{$image_field0}}:{{end}}" "${IMAGE_STATEFULSET_R1}:"
  kube::test::get_object_assert statefulset "{{range.items}}{{$container_len}}{{end}}" "1"
  # Rollback to revision 1000000 - should fail
  output_message=$(! kubectl rollout undo statefulset --to-revision=1000000 "${kube_flags[@]}" 2>&1)
  kube::test::if_has_string "${output_message}" "unable to find specified revision"
  kube::test::get_object_assert statefulset "{{range.items}}{{$image_field0}}:{{end}}" "${IMAGE_STATEFULSET_R1}:"
  kube::test::get_object_assert statefulset "{{range.items}}{{$container_len}}{{end}}" "1"
  # Rollback to last revision
  kubectl rollout undo statefulset "${kube_flags[@]}"
  kube::test::wait_object_assert statefulset "{{range.items}}{{$image_field0}}:{{end}}" "${IMAGE_STATEFULSET_R2}:"
  kube::test::wait_object_assert statefulset "{{range.items}}{{$image_field1}}:{{end}}" "${IMAGE_PAUSE_V2}:"
  kube::test::get_object_assert statefulset "{{range.items}}{{$container_len}}{{end}}" "2"
  # Clean up - delete newest configuration
  kubectl delete -f hack/testdata/rollingupdate-statefulset-rv2.yaml "${kube_flags[@]}"
  # Post-condition: no pods from statefulset controller
  wait-for-pods-with-label "app=nginx-statefulset" ""

  set +o nounset
  set +o errexit
}

run_multi_resources_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl(v1:multiple resources)"

  FILES="hack/testdata/multi-resource-yaml
  hack/testdata/multi-resource-list
  hack/testdata/multi-resource-json
  hack/testdata/multi-resource-rclist
  hack/testdata/multi-resource-svclist"
  YAML=".yaml"
  JSON=".json"
  for file in $FILES; do
    if [ -f $file$YAML ]
    then
      file=$file$YAML
      replace_file="${file%.yaml}-modify.yaml"
    else
      file=$file$JSON
      replace_file="${file%.json}-modify.json"
    fi

    has_svc=true
    has_rc=true
    two_rcs=false
    two_svcs=false
    if [[ "${file}" == *rclist* ]]; then
      has_svc=false
      two_rcs=true
    fi
    if [[ "${file}" == *svclist* ]]; then
      has_rc=false
      two_svcs=true
    fi

    ### Create, get, describe, replace, label, annotate, and then delete service nginxsvc and replication controller my-nginx from 5 types of files:
    ### 1) YAML, separated by ---; 2) JSON, with a List type; 3) JSON, with JSON object concatenation
    ### 4) JSON, with a ReplicationControllerList type; 5) JSON, with a ServiceList type
    echo "Testing with file ${file} and replace with file ${replace_file}"
    # Pre-condition: no service (other than default kubernetes services) or replication controller exists
    kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" ''
    kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" ''
    # Command
    kubectl create -f "${file}" "${kube_flags[@]}"
    # Post-condition: mock service (and mock2) exists
    if [ "$has_svc" = true ]; then
      if [ "$two_svcs" = true ]; then
        kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'mock:mock2:'
      else
        kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'mock:'
      fi
    fi
    # Post-condition: mock rc (and mock2) exists
    if [ "$has_rc" = true ]; then
      if [ "$two_rcs" = true ]; then
        kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" 'mock:mock2:'
      else
        kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" 'mock:'
      fi
    fi
    # Command
    kubectl get -f "${file}" "${kube_flags[@]}"
    # Command: watching multiple resources should return "not supported" error
    WATCH_ERROR_FILE="${KUBE_TEMP}/kubectl-watch-error"
    kubectl get -f "${file}" "${kube_flags[@]}" "--watch" 2> ${WATCH_ERROR_FILE} || true
    if ! grep -q "watch is only supported on individual resources and resource collections" "${WATCH_ERROR_FILE}"; then
      kube::log::error_exit "kubectl watch multiple resource returns unexpected error or non-error: $(cat ${WATCH_ERROR_FILE})" "1"
    fi
    kubectl describe -f "${file}" "${kube_flags[@]}"
    # Command
    kubectl replace -f $replace_file --force --cascade "${kube_flags[@]}"
    # Post-condition: mock service (and mock2) and mock rc (and mock2) are replaced
    if [ "$has_svc" = true ]; then
      kube::test::get_object_assert 'services mock' "{{${labels_field}.status}}" 'replaced'
      if [ "$two_svcs" = true ]; then
        kube::test::get_object_assert 'services mock2' "{{${labels_field}.status}}" 'replaced'
      fi
    fi
    if [ "$has_rc" = true ]; then
      kube::test::get_object_assert 'rc mock' "{{${labels_field}.status}}" 'replaced'
      if [ "$two_rcs" = true ]; then
        kube::test::get_object_assert 'rc mock2' "{{${labels_field}.status}}" 'replaced'
      fi
    fi
    # Command: kubectl edit multiple resources
    temp_editor="${KUBE_TEMP}/tmp-editor.sh"
    echo -e "#!/bin/bash\n$SED -i \"s/status\:\ replaced/status\:\ edited/g\" \$@" > "${temp_editor}"
    chmod +x "${temp_editor}"
    EDITOR="${temp_editor}" kubectl edit "${kube_flags[@]}" -f "${file}"
    # Post-condition: mock service (and mock2) and mock rc (and mock2) are edited
    if [ "$has_svc" = true ]; then
      kube::test::get_object_assert 'services mock' "{{${labels_field}.status}}" 'edited'
      if [ "$two_svcs" = true ]; then
        kube::test::get_object_assert 'services mock2' "{{${labels_field}.status}}" 'edited'
      fi
    fi
    if [ "$has_rc" = true ]; then
      kube::test::get_object_assert 'rc mock' "{{${labels_field}.status}}" 'edited'
      if [ "$two_rcs" = true ]; then
        kube::test::get_object_assert 'rc mock2' "{{${labels_field}.status}}" 'edited'
      fi
    fi
    # cleaning
    rm "${temp_editor}"
    # Command
    # We need to set --overwrite, because otherwise, if the first attempt to run "kubectl label"
    # fails on some, but not all, of the resources, retries will fail because it tries to modify
    # existing labels.
    kubectl-with-retry label -f $file labeled=true --overwrite "${kube_flags[@]}"
    # Post-condition: mock service and mock rc (and mock2) are labeled
    if [ "$has_svc" = true ]; then
      kube::test::get_object_assert 'services mock' "{{${labels_field}.labeled}}" 'true'
      if [ "$two_svcs" = true ]; then
        kube::test::get_object_assert 'services mock2' "{{${labels_field}.labeled}}" 'true'
      fi
    fi
    if [ "$has_rc" = true ]; then
      kube::test::get_object_assert 'rc mock' "{{${labels_field}.labeled}}" 'true'
      if [ "$two_rcs" = true ]; then
        kube::test::get_object_assert 'rc mock2' "{{${labels_field}.labeled}}" 'true'
      fi
    fi
    # Command
    # Command
    # We need to set --overwrite, because otherwise, if the first attempt to run "kubectl annotate"
    # fails on some, but not all, of the resources, retries will fail because it tries to modify
    # existing annotations.
    kubectl-with-retry annotate -f $file annotated=true --overwrite "${kube_flags[@]}"
    # Post-condition: mock service (and mock2) and mock rc (and mock2) are annotated
    if [ "$has_svc" = true ]; then
      kube::test::get_object_assert 'services mock' "{{${annotations_field}.annotated}}" 'true'
      if [ "$two_svcs" = true ]; then
        kube::test::get_object_assert 'services mock2' "{{${annotations_field}.annotated}}" 'true'
      fi
    fi
    if [ "$has_rc" = true ]; then
      kube::test::get_object_assert 'rc mock' "{{${annotations_field}.annotated}}" 'true'
      if [ "$two_rcs" = true ]; then
        kube::test::get_object_assert 'rc mock2' "{{${annotations_field}.annotated}}" 'true'
      fi
    fi
    # Cleanup resources created
    kubectl delete -f "${file}" "${kube_flags[@]}"
  done

  #############################
  # Multiple Resources via URL#
  #############################

  # Pre-condition: no service (other than default kubernetes services) or replication controller exists
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" ''
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" ''

  # Command
  kubectl create -f https://raw.githubusercontent.com/kubernetes/kubernetes/master/hack/testdata/multi-resource-yaml.yaml "${kube_flags[@]}"

  # Post-condition: service(mock) and rc(mock) exist
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'mock:'
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" 'mock:'

  # Clean up
  kubectl delete -f https://raw.githubusercontent.com/kubernetes/kubernetes/master/hack/testdata/multi-resource-yaml.yaml "${kube_flags[@]}"

  # Post-condition: no service (other than default kubernetes services) or replication controller exists
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" ''
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" ''

  set +o nounset
  set +o errexit
}

run_kubectl_config_set_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl(v1:config set)"

  kubectl config set-cluster test-cluster --server="https://does-not-work"

  # Get the api cert and add a comment to avoid flag parsing problems
  cert_data=$(echo "#Comment" && cat "${TMPDIR:-/tmp}/apiserver.crt")

  kubectl config set clusters.test-cluster.certificate-authority-data "$cert_data" --set-raw-bytes
  r_writen=$(kubectl config view --raw -o jsonpath='{.clusters[?(@.name == "test-cluster")].cluster.certificate-authority-data}')

  encoded=$(echo -n "$cert_data" | base64)
  kubectl config set clusters.test-cluster.certificate-authority-data "$encoded"
  e_writen=$(kubectl config view --raw -o jsonpath='{.clusters[?(@.name == "test-cluster")].cluster.certificate-authority-data}')

  test "$e_writen" == "$r_writen"

  set +o nounset
  set +o errexit
}

run_kubectl_local_proxy_tests() {
  set -o nounset
  set -o errexit

  kube::log::status "Testing kubectl local proxy"

  # Make sure the UI can be proxied
  start-proxy
  check-curl-proxy-code /ui 307
  check-curl-proxy-code /api/ui 404
  check-curl-proxy-code /api/v1/namespaces 200
  if kube::test::if_supports_resource "${metrics}" ; then
    check-curl-proxy-code /metrics 200
  fi
  if kube::test::if_supports_resource "${static}" ; then
    check-curl-proxy-code /static/ 200
  fi
  stop-proxy

  # Make sure the in-development api is accessible by default
  start-proxy
  check-curl-proxy-code /apis 200
  check-curl-proxy-code /apis/extensions/ 200
  stop-proxy

  # Custom paths let you see everything.
  start-proxy /custom
  check-curl-proxy-code /custom/ui 307
  if kube::test::if_supports_resource "${metrics}" ; then
    check-curl-proxy-code /custom/metrics 200
  fi
  check-curl-proxy-code /custom/api/v1/namespaces 200
  stop-proxy

  set +o nounset
  set +o errexit
}

run_RESTMapper_evaluation_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing RESTMapper"

  RESTMAPPER_ERROR_FILE="${KUBE_TEMP}/restmapper-error"

  ### Non-existent resource type should give a recognizeable error
  # Pre-condition: None
  # Command
  kubectl get "${kube_flags[@]}" unknownresourcetype 2>${RESTMAPPER_ERROR_FILE} || true
  if grep -q "the server doesn't have a resource type" "${RESTMAPPER_ERROR_FILE}"; then
    kube::log::status "\"kubectl get unknownresourcetype\" returns error as expected: $(cat ${RESTMAPPER_ERROR_FILE})"
  else
    kube::log::status "\"kubectl get unknownresourcetype\" returns unexpected error or non-error: $(cat ${RESTMAPPER_ERROR_FILE})"
    exit 1
  fi
  rm "${RESTMAPPER_ERROR_FILE}"
  # Post-condition: None

  set +o nounset
  set +o errexit
}

run_clusterroles_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing clusterroles"

  # make sure the server was properly bootstrapped with clusterroles and bindings
  kube::test::get_object_assert clusterroles/cluster-admin "{{.metadata.name}}" 'cluster-admin'
  kube::test::get_object_assert clusterrolebindings/cluster-admin "{{.metadata.name}}" 'cluster-admin'

  # test `kubectl create clusterrole`
  kubectl create "${kube_flags[@]}" clusterrole pod-admin --verb=* --resource=pods
  kube::test::get_object_assert clusterrole/pod-admin "{{range.rules}}{{range.verbs}}{{.}}:{{end}}{{end}}" '\*:'
  kube::test::get_object_assert clusterrole/pod-admin "{{range.rules}}{{range.resources}}{{.}}:{{end}}{{end}}" 'pods:'
  kube::test::get_object_assert clusterrole/pod-admin "{{range.rules}}{{range.apiGroups}}{{.}}:{{end}}{{end}}" ':'
  kubectl create "${kube_flags[@]}" clusterrole resource-reader --verb=get,list --resource=pods,deployments.extensions
  kube::test::get_object_assert clusterrole/resource-reader "{{range.rules}}{{range.verbs}}{{.}}:{{end}}{{end}}" 'get:list:get:list:'
  kube::test::get_object_assert clusterrole/resource-reader "{{range.rules}}{{range.resources}}{{.}}:{{end}}{{end}}" 'pods:deployments:'
  kube::test::get_object_assert clusterrole/resource-reader "{{range.rules}}{{range.apiGroups}}{{.}}:{{end}}{{end}}" ':extensions:'
  kubectl create "${kube_flags[@]}" clusterrole resourcename-reader --verb=get,list --resource=pods --resource-name=foo
  kube::test::get_object_assert clusterrole/resourcename-reader "{{range.rules}}{{range.verbs}}{{.}}:{{end}}{{end}}" 'get:list:'
  kube::test::get_object_assert clusterrole/resourcename-reader "{{range.rules}}{{range.resources}}{{.}}:{{end}}{{end}}" 'pods:'
  kube::test::get_object_assert clusterrole/resourcename-reader "{{range.rules}}{{range.apiGroups}}{{.}}:{{end}}{{end}}" ':'
  kube::test::get_object_assert clusterrole/resourcename-reader "{{range.rules}}{{range.resourceNames}}{{.}}:{{end}}{{end}}" 'foo:'
  kubectl create "${kube_flags[@]}" clusterrole url-reader --verb=get --non-resource-url=/logs/* --non-resource-url=/healthz/*
  kube::test::get_object_assert clusterrole/url-reader "{{range.rules}}{{range.verbs}}{{.}}:{{end}}{{end}}" 'get:'
  kube::test::get_object_assert clusterrole/url-reader "{{range.rules}}{{range.nonResourceURLs}}{{.}}:{{end}}{{end}}" '/logs/\*:/healthz/\*:'

  # test `kubectl create rolebinding/clusterrolebinding`
  # test `kubectl set subject rolebinding/clusterrolebinding`
  kubectl create "${kube_flags[@]}" clusterrolebinding super-admin --clusterrole=admin --user=super-admin
  kube::test::get_object_assert clusterrolebinding/super-admin "{{range.subjects}}{{.name}}:{{end}}" 'super-admin:'
  kubectl set subject "${kube_flags[@]}" clusterrolebinding super-admin --user=foo
  kube::test::get_object_assert clusterrolebinding/super-admin "{{range.subjects}}{{.name}}:{{end}}" 'super-admin:foo:'

  kubectl create "${kube_flags[@]}" clusterrolebinding super-group --clusterrole=admin --group=the-group
  kube::test::get_object_assert clusterrolebinding/super-group "{{range.subjects}}{{.name}}:{{end}}" 'the-group:'
  kubectl set subject "${kube_flags[@]}" clusterrolebinding super-group --group=foo
  kube::test::get_object_assert clusterrolebinding/super-group "{{range.subjects}}{{.name}}:{{end}}" 'the-group:foo:'

  kubectl create "${kube_flags[@]}" clusterrolebinding super-sa --clusterrole=admin --serviceaccount=otherns:sa-name
  kube::test::get_object_assert clusterrolebinding/super-sa "{{range.subjects}}{{.namespace}}:{{end}}" 'otherns:'
  kube::test::get_object_assert clusterrolebinding/super-sa "{{range.subjects}}{{.name}}:{{end}}" 'sa-name:'
  kubectl set subject "${kube_flags[@]}" clusterrolebinding super-sa --serviceaccount=otherfoo:foo
  kube::test::get_object_assert clusterrolebinding/super-sa "{{range.subjects}}{{.namespace}}:{{end}}" 'otherns:otherfoo:'
  kube::test::get_object_assert clusterrolebinding/super-sa "{{range.subjects}}{{.name}}:{{end}}" 'sa-name:foo:'

  kubectl create "${kube_flags[@]}" rolebinding admin --clusterrole=admin --user=default-admin
  kube::test::get_object_assert rolebinding/admin "{{range.subjects}}{{.name}}:{{end}}" 'default-admin:'
  kubectl set subject "${kube_flags[@]}" rolebinding admin --user=foo
  kube::test::get_object_assert rolebinding/admin "{{range.subjects}}{{.name}}:{{end}}" 'default-admin:foo:'

  kubectl create "${kube_flags[@]}" rolebinding localrole --role=localrole --group=the-group
  kube::test::get_object_assert rolebinding/localrole "{{range.subjects}}{{.name}}:{{end}}" 'the-group:'
  kubectl set subject "${kube_flags[@]}" rolebinding localrole --group=foo
  kube::test::get_object_assert rolebinding/localrole "{{range.subjects}}{{.name}}:{{end}}" 'the-group:foo:'

  kubectl create "${kube_flags[@]}" rolebinding sarole --role=localrole --serviceaccount=otherns:sa-name
  kube::test::get_object_assert rolebinding/sarole "{{range.subjects}}{{.namespace}}:{{end}}" 'otherns:'
  kube::test::get_object_assert rolebinding/sarole "{{range.subjects}}{{.name}}:{{end}}" 'sa-name:'
  kubectl set subject "${kube_flags[@]}" rolebinding sarole --serviceaccount=otherfoo:foo
  kube::test::get_object_assert rolebinding/sarole "{{range.subjects}}{{.namespace}}:{{end}}" 'otherns:otherfoo:'
  kube::test::get_object_assert rolebinding/sarole "{{range.subjects}}{{.name}}:{{end}}" 'sa-name:foo:'

  set +o nounset
  set +o errexit
}

run_role_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing role"

  # Create Role from command (only resource)
  kubectl create "${kube_flags[@]}" role pod-admin --verb=* --resource=pods
  kube::test::get_object_assert role/pod-admin "{{range.rules}}{{range.verbs}}{{.}}:{{end}}{{end}}" '\*:'
  kube::test::get_object_assert role/pod-admin "{{range.rules}}{{range.resources}}{{.}}:{{end}}{{end}}" 'pods:'
  kube::test::get_object_assert role/pod-admin "{{range.rules}}{{range.apiGroups}}{{.}}:{{end}}{{end}}" ':'
  output_message=$(! kubectl create "${kube_flags[@]}" role invalid-pod-admin --verb=* --resource=invalid-resource 2>&1)
  kube::test::if_has_string "${output_message}" "the server doesn't have a resource type \"invalid-resource\""
  # Create Role from command (resource + group)
  kubectl create "${kube_flags[@]}" role group-reader --verb=get,list --resource=deployments.extensions
  kube::test::get_object_assert role/group-reader "{{range.rules}}{{range.verbs}}{{.}}:{{end}}{{end}}" 'get:list:'
  kube::test::get_object_assert role/group-reader "{{range.rules}}{{range.resources}}{{.}}:{{end}}{{end}}" 'deployments:'
  kube::test::get_object_assert role/group-reader "{{range.rules}}{{range.apiGroups}}{{.}}:{{end}}{{end}}" 'extensions:'
  output_message=$(! kubectl create "${kube_flags[@]}" role invalid-group --verb=get,list --resource=deployments.invalid-group 2>&1)
  kube::test::if_has_string "${output_message}" "the server doesn't have a resource type \"deployments\" in group \"invalid-group\""
  # Create Role from command (resource / subresource)
  kubectl create "${kube_flags[@]}" role subresource-reader --verb=get,list --resource=pods/status
  kube::test::get_object_assert role/subresource-reader "{{range.rules}}{{range.verbs}}{{.}}:{{end}}{{end}}" 'get:list:'
  kube::test::get_object_assert role/subresource-reader "{{range.rules}}{{range.resources}}{{.}}:{{end}}{{end}}" 'pods/status:'
  kube::test::get_object_assert role/subresource-reader "{{range.rules}}{{range.apiGroups}}{{.}}:{{end}}{{end}}" ':'
  # Create Role from command (resource + group / subresource)
  kubectl create "${kube_flags[@]}" role group-subresource-reader --verb=get,list --resource=replicasets.extensions/scale
  kube::test::get_object_assert role/group-subresource-reader "{{range.rules}}{{range.verbs}}{{.}}:{{end}}{{end}}" 'get:list:'
  kube::test::get_object_assert role/group-subresource-reader "{{range.rules}}{{range.resources}}{{.}}:{{end}}{{end}}" 'replicasets/scale:'
  kube::test::get_object_assert role/group-subresource-reader "{{range.rules}}{{range.apiGroups}}{{.}}:{{end}}{{end}}" 'extensions:'
  output_message=$(! kubectl create "${kube_flags[@]}" role invalid-group --verb=get,list --resource=rs.invalid-group/scale 2>&1)
  kube::test::if_has_string "${output_message}" "the server doesn't have a resource type \"rs\" in group \"invalid-group\""
  # Create Role from command (resource + resourcename)
  kubectl create "${kube_flags[@]}" role resourcename-reader --verb=get,list --resource=pods --resource-name=foo
  kube::test::get_object_assert role/resourcename-reader "{{range.rules}}{{range.verbs}}{{.}}:{{end}}{{end}}" 'get:list:'
  kube::test::get_object_assert role/resourcename-reader "{{range.rules}}{{range.resources}}{{.}}:{{end}}{{end}}" 'pods:'
  kube::test::get_object_assert role/resourcename-reader "{{range.rules}}{{range.apiGroups}}{{.}}:{{end}}{{end}}" ':'
  kube::test::get_object_assert role/resourcename-reader "{{range.rules}}{{range.resourceNames}}{{.}}:{{end}}{{end}}" 'foo:'
  # Create Role from command (multi-resources)
  kubectl create "${kube_flags[@]}" role resource-reader --verb=get,list --resource=pods/status,deployments.extensions
  kube::test::get_object_assert role/resource-reader "{{range.rules}}{{range.verbs}}{{.}}:{{end}}{{end}}" 'get:list:get:list:'
  kube::test::get_object_assert role/resource-reader "{{range.rules}}{{range.resources}}{{.}}:{{end}}{{end}}" 'pods/status:deployments:'
  kube::test::get_object_assert role/resource-reader "{{range.rules}}{{range.apiGroups}}{{.}}:{{end}}{{end}}" ':extensions:'

  set +o nounset
  set +o errexit
}

run_assert_short_name_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing assert short name"

  kube::log::status "Testing propagation of short names for resources"
  output_message=$(kubectl get --raw=/api/v1)

  ## test if a short name is exported during discovery
  kube::test::if_has_string "${output_message}" '{"name":"configmaps","singularName":"","namespaced":true,"kind":"ConfigMap","verbs":\["create","delete","deletecollection","get","list","patch","update","watch"\],"shortNames":\["cm"\]}'

  set +o nounset
  set +o errexit
}

run_assert_categories_tests() {
  set -o nounset
  set -o errexit

  kube::log::status "Testing propagation of categories for resources"
  output_message=$(kubectl get --raw=/api/v1 | grep -Po '"name":"pods".*?}')
  kube::test::if_has_string "${output_message}" '"categories":\["all"\]'

  set +o nounset
  set +o errexit
}

run_kubectl_create_error_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl create with error"

  # Passing no arguments to create is an error
  ! kubectl create

  ## kubectl create should not panic on empty string lists in a template
  ERROR_FILE="${KUBE_TEMP}/validation-error"
  kubectl create -f hack/testdata/invalid-rc-with-empty-args.yaml "${kube_flags[@]}" 2> "${ERROR_FILE}" || true
  # Post-condition: should get an error reporting the empty string
  if grep -q "unknown object type \"nil\" in ReplicationController" "${ERROR_FILE}"; then
    kube::log::status "\"kubectl create with empty string list returns error as expected: $(cat ${ERROR_FILE})"
  else
    kube::log::status "\"kubectl create with empty string list returns unexpected error or non-error: $(cat ${ERROR_FILE})"
    exit 1
  fi
  rm "${ERROR_FILE}"

  set +o nounset
  set +o errexit
}

run_cmd_with_img_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing cmd with image"

  # Test that a valid image reference value is provided as the value of --image in `kubectl run <name> --image`
  output_message=$(kubectl run test1 --image=validname)
  kube::test::if_has_string "${output_message}" 'deployment "test1" created'
  kubectl delete deployments test1
  # test invalid image name
  output_message=$(! kubectl run test2 --image=InvalidImageName 2>&1)
  kube::test::if_has_string "${output_message}" 'error: Invalid image name "InvalidImageName": invalid reference format'

  set +o nounset
  set +o errexit
}

run_client_config_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing client config"

  # Command
  # Pre-condition: kubeconfig "missing" is not a file or directory
  output_message=$(! kubectl get pod --context="" --kubeconfig=missing 2>&1)
  kube::test::if_has_string "${output_message}" "missing: no such file or directory"

  # Pre-condition: kubeconfig "missing" is not a file or directory
  # Command
  output_message=$(! kubectl get pod --user="" --kubeconfig=missing 2>&1)
  # Post-condition: --user contains a valid / empty value, missing config file returns error
  kube::test::if_has_string "${output_message}" "missing: no such file or directory"
  # Command
  output_message=$(! kubectl get pod --cluster="" --kubeconfig=missing 2>&1)
  # Post-condition: --cluster contains a "valid" value, missing config file returns error
  kube::test::if_has_string "${output_message}" "missing: no such file or directory"

  # Pre-condition: context "missing-context" does not exist
  # Command
  output_message=$(! kubectl get pod --context="missing-context" 2>&1)
  kube::test::if_has_string "${output_message}" 'context "missing-context" does not exist'
  # Post-condition: invalid or missing context returns error

  # Pre-condition: cluster "missing-cluster" does not exist
  # Command
  output_message=$(! kubectl get pod --cluster="missing-cluster" 2>&1)
  kube::test::if_has_string "${output_message}" 'cluster "missing-cluster" does not exist'
  # Post-condition: invalid or missing cluster returns error

  # Pre-condition: user "missing-user" does not exist
  # Command
  output_message=$(! kubectl get pod --user="missing-user" 2>&1)
  kube::test::if_has_string "${output_message}" 'auth info "missing-user" does not exist'
  # Post-condition: invalid or missing user returns error

  # test invalid config
  kubectl config view | sed -E "s/apiVersion: .*/apiVersion: v-1/g" > "${TMPDIR:-/tmp}"/newconfig.yaml
  output_message=$(! "${KUBE_OUTPUT_HOSTBIN}/kubectl" get pods --context="" --user="" --kubeconfig="${TMPDIR:-/tmp}"/newconfig.yaml 2>&1)
  kube::test::if_has_string "${output_message}" "Error loading config file"

  output_message=$(! kubectl get pod --kubeconfig=missing-config 2>&1)
  kube::test::if_has_string "${output_message}" 'no such file or directory'

  set +o nounset
  set +o errexit
}

run_service_accounts_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing service accounts"

  ### Create a new namespace
  # Pre-condition: the test-service-accounts namespace does not exist
  kube::test::get_object_assert 'namespaces' '{{range.items}}{{ if eq $id_field \"test-service-accounts\" }}found{{end}}{{end}}:' ':'
  # Command
  kubectl create namespace test-service-accounts
  # Post-condition: namespace 'test-service-accounts' is created.
  kube::test::get_object_assert 'namespaces/test-service-accounts' "{{$id_field}}" 'test-service-accounts'

  ### Create a service account in a specific namespace
  # Command
  kubectl create serviceaccount test-service-account --namespace=test-service-accounts
  # Post-condition: secret exists and has expected values
  kube::test::get_object_assert 'serviceaccount/test-service-account --namespace=test-service-accounts' "{{$id_field}}" 'test-service-account'
  # Clean-up
  kubectl delete serviceaccount test-service-account --namespace=test-service-accounts
  # Clean up
  kubectl delete namespace test-service-accounts

  set +o nounset
  set +o errexit
}

run_pod_templates_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing pod templates"

  ### Create PODTEMPLATE
  # Pre-condition: no PODTEMPLATE
  kube::test::get_object_assert podtemplates "{{range.items}}{{.metadata.name}}:{{end}}" ''
  # Command
  kubectl create -f test/fixtures/doc-yaml/user-guide/walkthrough/podtemplate.json "${kube_flags[@]}"
  # Post-condition: nginx PODTEMPLATE is available
  kube::test::get_object_assert podtemplates "{{range.items}}{{.metadata.name}}:{{end}}" 'nginx:'

  ### Printing pod templates works
  kubectl get podtemplates "${kube_flags[@]}"
  [[ "$(kubectl get podtemplates -o yaml "${kube_flags[@]}" | grep nginx)" ]]

  ### Delete nginx pod template by name
  # Pre-condition: nginx pod template is available
  kube::test::get_object_assert podtemplates "{{range.items}}{{.metadata.name}}:{{end}}" 'nginx:'
  # Command
  kubectl delete podtemplate nginx "${kube_flags[@]}"
  # Post-condition: No templates exist
  kube::test::get_object_assert podtemplate "{{range.items}}{{.metadata.name}}:{{end}}" ''

  set +o nounset
  set +o errexit
}

run_stateful_set_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl(v1:statefulsets)"

  ### Create and stop statefulset, make sure it doesn't leak pods
  # Pre-condition: no statefulset exists
  kube::test::get_object_assert statefulset "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command: create statefulset
  kubectl create -f hack/testdata/rollingupdate-statefulset.yaml "${kube_flags[@]}"

  ### Scale statefulset test with current-replicas and replicas
  # Pre-condition: 0 replicas
  kube::test::get_object_assert 'statefulset nginx' "{{$statefulset_replicas_field}}" '0'
  kube::test::wait_object_assert 'statefulset nginx' "{{$statefulset_observed_generation}}" '1'
  # Command: Scale up
  kubectl scale --current-replicas=0 --replicas=1 statefulset nginx "${kube_flags[@]}"
  # Post-condition: 1 replica, named nginx-0
  kube::test::get_object_assert 'statefulset nginx' "{{$statefulset_replicas_field}}" '1'
  kube::test::wait_object_assert 'statefulset nginx' "{{$statefulset_observed_generation}}" '2'
  # Typically we'd wait and confirm that N>1 replicas are up, but this framework
  # doesn't start  the scheduler, so pet-0 will block all others.
  # TODO: test robust scaling in an e2e.
  wait-for-pods-with-label "app=nginx-statefulset" "nginx-0"

  ### Clean up
  kubectl delete -f hack/testdata/rollingupdate-statefulset.yaml "${kube_flags[@]}"
  # Post-condition: no pods from statefulset controller
  wait-for-pods-with-label "app=nginx-statefulset" ""

  set +o nounset
  set +o errexit

}

run_lists_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl(v1:lists)"

  ### Create a List with objects from multiple versions
  # Command
  kubectl create -f hack/testdata/list.yaml "${kube_flags[@]}"

  ### Delete the List with objects from multiple versions
  # Command
  kubectl delete service/list-service-test deployment/list-deployment-test

  set +o nounset
  set +o errexit
}

run_persistent_volumes_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing persistent volumes"

  ### Create and delete persistent volume examples
  # Pre-condition: no persistent volumes currently exist
  kube::test::get_object_assert pv "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f test/fixtures/doc-yaml/user-guide/persistent-volumes/volumes/local-01.yaml "${kube_flags[@]}"
  kube::test::get_object_assert pv "{{range.items}}{{$id_field}}:{{end}}" 'pv0001:'
  kubectl delete pv pv0001 "${kube_flags[@]}"
  kubectl create -f test/fixtures/doc-yaml/user-guide/persistent-volumes/volumes/local-02.yaml "${kube_flags[@]}"
  kube::test::get_object_assert pv "{{range.items}}{{$id_field}}:{{end}}" 'pv0002:'
  kubectl delete pv pv0002 "${kube_flags[@]}"
  kubectl create -f test/fixtures/doc-yaml/user-guide/persistent-volumes/volumes/gce.yaml "${kube_flags[@]}"
  kube::test::get_object_assert pv "{{range.items}}{{$id_field}}:{{end}}" 'pv0003:'
  kubectl delete pv pv0003 "${kube_flags[@]}"
  # Post-condition: no PVs
  kube::test::get_object_assert pv "{{range.items}}{{$id_field}}:{{end}}" ''

  set +o nounset
  set +o errexit
}

run_persistent_volume_claims_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing persistent volumes claims"

  ### Create and delete persistent volume claim examples
  # Pre-condition: no persistent volume claims currently exist
  kube::test::get_object_assert pvc "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f test/fixtures/doc-yaml/user-guide/persistent-volumes/claims/claim-01.yaml "${kube_flags[@]}"
  kube::test::get_object_assert pvc "{{range.items}}{{$id_field}}:{{end}}" 'myclaim-1:'
  kubectl delete pvc myclaim-1 "${kube_flags[@]}"

  kubectl create -f test/fixtures/doc-yaml/user-guide/persistent-volumes/claims/claim-02.yaml "${kube_flags[@]}"
  kube::test::get_object_assert pvc "{{range.items}}{{$id_field}}:{{end}}" 'myclaim-2:'
  kubectl delete pvc myclaim-2 "${kube_flags[@]}"

  kubectl create -f test/fixtures/doc-yaml/user-guide/persistent-volumes/claims/claim-03.json "${kube_flags[@]}"
  kube::test::get_object_assert pvc "{{range.items}}{{$id_field}}:{{end}}" 'myclaim-3:'
  kubectl delete pvc myclaim-3 "${kube_flags[@]}"
  # Post-condition: no PVCs
  kube::test::get_object_assert pvc "{{range.items}}{{$id_field}}:{{end}}" ''

  set +o nounset
  set +o errexit
}

run_storage_class_tests() {
  set -o nounset
  set -o errexit

  kube::log::status "Testing storage class"

  ### Create and delete storage class
  # Pre-condition: no storage classes currently exist
  kube::test::get_object_assert storageclass "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f - "${kube_flags[@]}" << __EOF__
{
  "kind": "StorageClass",
  "apiVersion": "storage.k8s.io/v1",
  "metadata": {
  "name": "storage-class-name"
  },
  "provisioner": "kubernetes.io/fake-provisioner-type",
  "parameters": {
  "zone":"us-east-1b",
  "type":"ssd"
  }
}
__EOF__
  kube::test::get_object_assert storageclass "{{range.items}}{{$id_field}}:{{end}}" 'storage-class-name:'
  kube::test::get_object_assert sc "{{range.items}}{{$id_field}}:{{end}}" 'storage-class-name:'
  kubectl delete storageclass storage-class-name "${kube_flags[@]}"
  # Post-condition: no storage classes
  kube::test::get_object_assert storageclass "{{range.items}}{{$id_field}}:{{end}}" ''

  set +o nounset
  set +o errexit

}

run_nodes_tests() {
  set -o nounset
  set -o errexit

  kube::log::status "Testing kubectl(v1:nodes)"

  kube::test::get_object_assert nodes "{{range.items}}{{$id_field}}:{{end}}" '127.0.0.1:'

  kube::test::describe_object_assert nodes "127.0.0.1" "Name:" "Labels:" "CreationTimestamp:" "Conditions:" "Addresses:" "Capacity:" "Pods:"
  # Describe command should print events information by default
  kube::test::describe_object_events_assert nodes "127.0.0.1"
  # Describe command should not print events information when show-events=false
  kube::test::describe_object_events_assert nodes "127.0.0.1" false
  # Describe command should print events information when show-events=true
  kube::test::describe_object_events_assert nodes "127.0.0.1" true
  # Describe command (resource only) should print detailed information
  kube::test::describe_resource_assert nodes "Name:" "Labels:" "CreationTimestamp:" "Conditions:" "Addresses:" "Capacity:" "Pods:"
  # Describe command should print events information by default
  kube::test::describe_resource_events_assert nodes
  # Describe command should not print events information when show-events=false
  kube::test::describe_resource_events_assert nodes false
  # Describe command should print events information when show-events=true
  kube::test::describe_resource_events_assert nodes true

  ### kubectl patch update can mark node unschedulable
  # Pre-condition: node is schedulable
  kube::test::get_object_assert "nodes 127.0.0.1" "{{.spec.unschedulable}}" '<no value>'
  kubectl patch "${kube_flags[@]}" nodes "127.0.0.1" -p='{"spec":{"unschedulable":true}}'
  # Post-condition: node is unschedulable
  kube::test::get_object_assert "nodes 127.0.0.1" "{{.spec.unschedulable}}" 'true'
  kubectl patch "${kube_flags[@]}" nodes "127.0.0.1" -p='{"spec":{"unschedulable":null}}'
  # Post-condition: node is schedulable
  kube::test::get_object_assert "nodes 127.0.0.1" "{{.spec.unschedulable}}" '<no value>'

  # check webhook token authentication endpoint, kubectl doesn't actually display the returned object so this isn't super useful
  # but it proves that works
  kubectl create -f test/fixtures/pkg/kubectl/cmd/create/tokenreview-v1beta1.json --validate=false
  kubectl create -f test/fixtures/pkg/kubectl/cmd/create/tokenreview-v1.json --validate=false

  set +o nounset
  set +o errexit
}

run_authorization_tests() {
  set -o nounset
  set -o errexit

  kube::log::status "Testing authorization"

  # check remote authorization endpoint, kubectl doesn't actually display the returned object so this isn't super useful
  # but it proves that works
  kubectl create -f test/fixtures/pkg/kubectl/cmd/create/sar-v1.json --validate=false
  kubectl create -f test/fixtures/pkg/kubectl/cmd/create/sar-v1beta1.json --validate=false

  SAR_RESULT_FILE="${KUBE_TEMP}/sar-result.json"
  curl -k -H "Content-Type:" http://localhost:8080/apis/authorization.k8s.io/v1beta1/subjectaccessreviews -XPOST -d @test/fixtures/pkg/kubectl/cmd/create/sar-v1beta1.json > "${SAR_RESULT_FILE}"
  if grep -q '"allowed": true' "${SAR_RESULT_FILE}"; then
    kube::log::status "\"authorization.k8s.io/subjectaccessreviews\" returns as expected: $(cat "${SAR_RESULT_FILE}")"
  else
    kube::log::status "\"authorization.k8s.io/subjectaccessreviews\" does not return as expected: $(cat "${SAR_RESULT_FILE}")"
    exit 1
  fi
  rm "${SAR_RESULT_FILE}"

  SAR_RESULT_FILE="${KUBE_TEMP}/sar-result.json"
  curl -k -H "Content-Type:" http://localhost:8080/apis/authorization.k8s.io/v1/subjectaccessreviews -XPOST -d @test/fixtures/pkg/kubectl/cmd/create/sar-v1.json > "${SAR_RESULT_FILE}"
  if grep -q '"allowed": true' "${SAR_RESULT_FILE}"; then
    kube::log::status "\"authorization.k8s.io/subjectaccessreviews\" returns as expected: $(cat "${SAR_RESULT_FILE}")"
  else
    kube::log::status "\"authorization.k8s.io/subjectaccessreviews\" does not return as expected: $(cat "${SAR_RESULT_FILE}")"
    exit 1
  fi
  rm "${SAR_RESULT_FILE}"

  set +o nounset
  set +o errexit
}

run_retrieve_multiple_tests() {
  set -o nounset
  set -o errexit

  # switch back to the default namespace
  kubectl config set-context "${CONTEXT}" --namespace=""
  kube::log::status "Testing kubectl(v1:multiget)"
  kube::test::get_object_assert 'nodes/127.0.0.1 service/kubernetes' "{{range.items}}{{$id_field}}:{{end}}" '127.0.0.1:kubernetes:'

  set +o nounset
  set +o errexit
}

run_resource_aliasing_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing resource aliasing"
  kubectl create -f examples/storage/cassandra/cassandra-controller.yaml "${kube_flags[@]}"
  kubectl create -f examples/storage/cassandra/cassandra-service.yaml "${kube_flags[@]}"

  object="all -l'app=cassandra'"
  request="{{range.items}}{{range .metadata.labels}}{{.}}:{{end}}{{end}}"

  # all 4 cassandra's might not be in the request immediately...
  kube::test::get_object_assert "$object" "$request" 'cassandra:cassandra:cassandra:cassandra:' || \
  kube::test::get_object_assert "$object" "$request" 'cassandra:cassandra:cassandra:' || \
  kube::test::get_object_assert "$object" "$request" 'cassandra:cassandra:'

  kubectl delete all -l app=cassandra "${kube_flags[@]}"

  set +o nounset
  set +o errexit
}

run_kubectl_explain_tests() {
  set -o nounset
  set -o errexit

  kube::log::status "Testing kubectl(v1:explain)"
  kubectl explain pods
  # shortcuts work
  kubectl explain po
  kubectl explain po.status.message

  set +o nounset
  set +o errexit
}

run_swagger_tests() {
  set -o nounset
  set -o errexit

  kube::log::status "Testing swagger"

  # Verify schema
  file="${KUBE_TEMP}/schema-v1.json"
  curl -s "http://127.0.0.1:${API_PORT}/swaggerapi/api/v1" > "${file}"
  [[ "$(grep "list of returned" "${file}")" ]]
  [[ "$(grep "List of services" "${file}")" ]]
  [[ "$(grep "Watch for changes to the described resources" "${file}")" ]]

  set +o nounset
  set +o errexit
}

run_kubectl_sort_by_tests() {
  set -o nounset
  set -o errexit

  kube::log::status "Testing kubectl --sort-by"

  ### sort-by should not panic if no pod exists
  # Pre-condition: no POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl get pods --sort-by="{metadata.name}"
  kubectl get pods --sort-by="{metadata.creationTimestamp}"

  ### sort-by should works if pod exists
  # Create POD
  # Pre-condition: no POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create "${kube_flags[@]}" -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml
  # Post-condition: valid-pod is created
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'
  # Check output of sort-by
  output_message=$(kubectl get pods --sort-by="{metadata.name}")
  kube::test::if_has_string "${output_message}" "valid-pod"
  ### Clean up
  # Pre-condition: valid-pod exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'
  # Command
  kubectl delete "${kube_flags[@]}" pod valid-pod --grace-period=0 --force
  # Post-condition: valid-pod doesn't exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''

  ### sort-by should works by sorting by name
  # Create three PODs
  # Pre-condition: no POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create "${kube_flags[@]}" -f hack/testdata/sorted-pods/sorted-pod1.yaml
  # Post-condition: sorted-pod1 is created
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'sorted-pod1:'
  # Command
  kubectl create "${kube_flags[@]}" -f hack/testdata/sorted-pods/sorted-pod2.yaml
  # Post-condition: sorted-pod1 is created
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'sorted-pod1:sorted-pod2:'
  # Command
  kubectl create "${kube_flags[@]}" -f hack/testdata/sorted-pods/sorted-pod3.yaml
  # Post-condition: sorted-pod1 is created
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'sorted-pod1:sorted-pod2:sorted-pod3:'

  # Check output of sort-by '{metadata.name}'
  output_message=$(kubectl get pods --sort-by="{metadata.name}")
  kube::test::if_sort_by_has_correct_order "${output_message}" "sorted-pod1:sorted-pod2:sorted-pod3:"

  # Check output of sort-by '{metadata.labels.name}'
  output_message=$(kubectl get pods --sort-by="{metadata.labels.name}")
  kube::test::if_sort_by_has_correct_order "${output_message}" "sorted-pod3:sorted-pod2:sorted-pod1:"

  ### Clean up
  # Pre-condition: valid-pod exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'sorted-pod1:sorted-pod2:sorted-pod3:'
  # Command
  kubectl delete "${kube_flags[@]}" pod --grace-period=0 --force --all
  # Post-condition: valid-pod doesn't exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''

  set +o nounset
  set +o errexit
}

run_kubectl_all_namespace_tests() {
  set -o nounset
  set -o errexit

  kube::log::status "Testing kubectl --all-namespace"

  # Pre-condition: the "default" namespace exists
  kube::test::get_object_assert namespaces "{{range.items}}{{if eq $id_field \\\"default\\\"}}{{$id_field}}:{{end}}{{end}}" 'default:'

  ### Create POD
  # Pre-condition: no POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create "${kube_flags[@]}" -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml
  # Post-condition: valid-pod is created
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'

  ### Verify a specific namespace is ignored when all-namespaces is provided
  # Command
  kubectl get pods --all-namespaces --namespace=default

  ### Clean up
  # Pre-condition: valid-pod exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'
  # Command
  kubectl delete "${kube_flags[@]}" pod valid-pod --grace-period=0 --force
  # Post-condition: valid-pod doesn't exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''

  set +o nounset
  set +o errexit
}

run_certificates_tests() {
  set -o nounset
  set -o errexit

  kube::log::status "Testing certificates"

  # approve
  kubectl create -f hack/testdata/csr.yml "${kube_flags[@]}"
  kube::test::get_object_assert 'csr/foo' '{{range.status.conditions}}{{.type}}{{end}}' ''
  kubectl certificate approve foo "${kube_flags[@]}"
  kubectl get csr "${kube_flags[@]}" -o json
  kube::test::get_object_assert 'csr/foo' '{{range.status.conditions}}{{.type}}{{end}}' 'Approved'
  kubectl delete -f hack/testdata/csr.yml "${kube_flags[@]}"
  kube::test::get_object_assert csr "{{range.items}}{{$id_field}}{{end}}" ''

  kubectl create -f hack/testdata/csr.yml "${kube_flags[@]}"
  kube::test::get_object_assert 'csr/foo' '{{range.status.conditions}}{{.type}}{{end}}' ''
  kubectl certificate approve -f hack/testdata/csr.yml "${kube_flags[@]}"
  kubectl get csr "${kube_flags[@]}" -o json
  kube::test::get_object_assert 'csr/foo' '{{range.status.conditions}}{{.type}}{{end}}' 'Approved'
  kubectl delete -f hack/testdata/csr.yml "${kube_flags[@]}"
  kube::test::get_object_assert csr "{{range.items}}{{$id_field}}{{end}}" ''

  # deny
  kubectl create -f hack/testdata/csr.yml "${kube_flags[@]}"
  kube::test::get_object_assert 'csr/foo' '{{range.status.conditions}}{{.type}}{{end}}' ''
  kubectl certificate deny foo "${kube_flags[@]}"
  kubectl get csr "${kube_flags[@]}" -o json
  kube::test::get_object_assert 'csr/foo' '{{range.status.conditions}}{{.type}}{{end}}' 'Denied'
  kubectl delete -f hack/testdata/csr.yml "${kube_flags[@]}"
  kube::test::get_object_assert csr "{{range.items}}{{$id_field}}{{end}}" ''

  kubectl create -f hack/testdata/csr.yml "${kube_flags[@]}"
  kube::test::get_object_assert 'csr/foo' '{{range.status.conditions}}{{.type}}{{end}}' ''
  kubectl certificate deny -f hack/testdata/csr.yml "${kube_flags[@]}"
  kubectl get csr "${kube_flags[@]}" -o json
  kube::test::get_object_assert 'csr/foo' '{{range.status.conditions}}{{.type}}{{end}}' 'Denied'
  kubectl delete -f hack/testdata/csr.yml "${kube_flags[@]}"
  kube::test::get_object_assert csr "{{range.items}}{{$id_field}}{{end}}" ''

  set +o nounset
  set +o errexit
}

run_plugins_tests() {
  set -o nounset
  set -o errexit

  kube::log::status "Testing kubectl plugins"

  # top-level plugin command
  output_message=$(KUBECTL_PLUGINS_PATH=test/fixtures/pkg/kubectl/plugins kubectl -h 2>&1)
  kube::test::if_has_string "${output_message}" 'plugin\s\+Runs a command-line plugin'

  # no plugins
  output_message=$(! kubectl plugin 2>&1)
  kube::test::if_has_string "${output_message}" 'no plugins installed'

  # single plugins path
  output_message=$(KUBECTL_PLUGINS_PATH=test/fixtures/pkg/kubectl/plugins kubectl plugin 2>&1)
  kube::test::if_has_string "${output_message}" 'echo\s\+Echoes for test-cmd'
  kube::test::if_has_string "${output_message}" 'get\s\+The wonderful new plugin-based get!'
  kube::test::if_has_string "${output_message}" 'error\s\+The tremendous plugin that always fails!'
  kube::test::if_has_not_string "${output_message}" 'The hello plugin'
  kube::test::if_has_not_string "${output_message}" 'Incomplete plugin'
  kube::test::if_has_not_string "${output_message}" 'no plugins installed'

  # multiple plugins path
  output_message=$(KUBECTL_PLUGINS_PATH=test/fixtures/pkg/kubectl/plugins/:test/fixtures/pkg/kubectl/plugins2/ kubectl plugin -h 2>&1)
  kube::test::if_has_string "${output_message}" 'echo\s\+Echoes for test-cmd'
  kube::test::if_has_string "${output_message}" 'get\s\+The wonderful new plugin-based get!'
  kube::test::if_has_string "${output_message}" 'error\s\+The tremendous plugin that always fails!'
  kube::test::if_has_string "${output_message}" 'hello\s\+The hello plugin'
  kube::test::if_has_not_string "${output_message}" 'Incomplete plugin'

  # don't override existing commands
  output_message=$(KUBECTL_PLUGINS_PATH=test/fixtures/pkg/kubectl/plugins/:test/fixtures/pkg/kubectl/plugins2/ kubectl get -h 2>&1)
  kube::test::if_has_string "${output_message}" 'Display one or many resources'
  kube::test::if_has_not_string "$output_message{output_message}" 'The wonderful new plugin-based get'

  # plugin help
  output_message=$(KUBECTL_PLUGINS_PATH=test/fixtures/pkg/kubectl/plugins/:test/fixtures/pkg/kubectl/plugins2/ kubectl plugin hello -h 2>&1)
  kube::test::if_has_string "${output_message}" 'The hello plugin is a new plugin used by test-cmd to test multiple plugin locations.'
  kube::test::if_has_string "${output_message}" 'Usage:'

  # run plugin
  output_message=$(KUBECTL_PLUGINS_PATH=test/fixtures/pkg/kubectl/plugins/:test/fixtures/pkg/kubectl/plugins2/ kubectl plugin hello 2>&1)
  kube::test::if_has_string "${output_message}" '#hello#'
  output_message=$(KUBECTL_PLUGINS_PATH=test/fixtures/pkg/kubectl/plugins/:test/fixtures/pkg/kubectl/plugins2/ kubectl plugin echo 2>&1)
  kube::test::if_has_string "${output_message}" 'This plugin works!'
  output_message=$(! KUBECTL_PLUGINS_PATH=test/fixtures/pkg/kubectl/plugins/ kubectl plugin hello 2>&1)
  kube::test::if_has_string "${output_message}" 'unknown command'
  output_message=$(! KUBECTL_PLUGINS_PATH=test/fixtures/pkg/kubectl/plugins/ kubectl plugin error 2>&1)
  kube::test::if_has_string "${output_message}" 'error: exit status 1'

  # plugin tree
  output_message=$(KUBECTL_PLUGINS_PATH=test/fixtures/pkg/kubectl/plugins kubectl plugin tree 2>&1)
  kube::test::if_has_string "${output_message}" 'Plugin with a tree of commands'
  kube::test::if_has_string "${output_message}" 'child1\s\+The first child of a tree'
  kube::test::if_has_string "${output_message}" 'child2\s\+The second child of a tree'
  kube::test::if_has_string "${output_message}" 'child3\s\+The third child of a tree'
  output_message=$(KUBECTL_PLUGINS_PATH=test/fixtures/pkg/kubectl/plugins kubectl plugin tree child1 --help 2>&1)
  kube::test::if_has_string "${output_message}" 'The first child of a tree'
  kube::test::if_has_not_string "${output_message}" 'The second child'
  kube::test::if_has_not_string "${output_message}" 'child2'
  output_message=$(KUBECTL_PLUGINS_PATH=test/fixtures/pkg/kubectl/plugins kubectl plugin tree child1 2>&1)
  kube::test::if_has_string "${output_message}" 'child one'
  kube::test::if_has_not_string "${output_message}" 'child1'
  kube::test::if_has_not_string "${output_message}" 'The first child'

  # plugin env
  output_message=$(KUBECTL_PLUGINS_PATH=test/fixtures/pkg/kubectl/plugins kubectl plugin env -h 2>&1)
  kube::test::if_has_string "${output_message}" "This is a flag 1"
  kube::test::if_has_string "${output_message}" "This is a flag 2"
  kube::test::if_has_string "${output_message}" "This is a flag 3"
  output_message=$(KUBECTL_PLUGINS_PATH=test/fixtures/pkg/kubectl/plugins kubectl plugin env --test1=value1 -t value2 2>&1)
  kube::test::if_has_string "${output_message}" 'KUBECTL_PLUGINS_CURRENT_NAMESPACE'
  kube::test::if_has_string "${output_message}" 'KUBECTL_PLUGINS_CALLER'
  kube::test::if_has_string "${output_message}" 'KUBECTL_PLUGINS_DESCRIPTOR_COMMAND=./env.sh'
  kube::test::if_has_string "${output_message}" 'KUBECTL_PLUGINS_DESCRIPTOR_SHORT_DESC=The plugin envs plugin'
  kube::test::if_has_string "${output_message}" 'KUBECTL_PLUGINS_GLOBAL_FLAG_KUBECONFIG'
  kube::test::if_has_string "${output_message}" 'KUBECTL_PLUGINS_GLOBAL_FLAG_REQUEST_TIMEOUT=0'
  kube::test::if_has_string "${output_message}" 'KUBECTL_PLUGINS_LOCAL_FLAG_TEST1=value1'
  kube::test::if_has_string "${output_message}" 'KUBECTL_PLUGINS_LOCAL_FLAG_TEST2=value2'
  kube::test::if_has_string "${output_message}" 'KUBECTL_PLUGINS_LOCAL_FLAG_TEST3=default'

  set +o nounset
  set +o errexit
}

run_impersonation_tests() {
  set -o nounset
  set -o errexit

  kube::log::status "Testing impersonation"

  output_message=$(! kubectl get pods "${kube_flags_with_token[@]}" --as-group=foo 2>&1)
  kube::test::if_has_string "${output_message}" 'without impersonating a user'

  if kube::test::if_supports_resource "${csr}" ; then
    # --as
    kubectl create -f hack/testdata/csr.yml "${kube_flags_with_token[@]}" --as=user1
    kube::test::get_object_assert 'csr/foo' '{{.spec.username}}' 'user1'
    kube::test::get_object_assert 'csr/foo' '{{range .spec.groups}}{{.}}{{end}}' 'system:authenticated'
    kubectl delete -f hack/testdata/csr.yml "${kube_flags_with_token[@]}"

    # --as-group
    kubectl create -f hack/testdata/csr.yml "${kube_flags_with_token[@]}" --as=user1 --as-group=group2 --as-group=group1 --as-group=,,,chameleon
    kube::test::get_object_assert 'csr/foo' '{{len .spec.groups}}' '3'
    kube::test::get_object_assert 'csr/foo' '{{range .spec.groups}}{{.}} {{end}}' 'group2 group1 ,,,chameleon '
    kubectl delete -f hack/testdata/csr.yml "${kube_flags_with_token[@]}"
  fi

  set +o nounset
  set +o errexit
}
# Runs all kubectl tests.
# Requires an env var SUPPORTED_RESOURCES which is a comma separated list of
# resources for which tests should be run.
runTests() {
  foundError="False"

  if [ -z "${SUPPORTED_RESOURCES:-}" ]; then
    echo "Need to set SUPPORTED_RESOURCES env var. It is a list of resources that are supported and hence should be tested. Set it to (*) to test all resources"
    exit 1
  fi
  kube::log::status "Checking kubectl version"
  kubectl version

  # use timestamp as the name of namespace because increasing the variable inside subshell
  # does not affect the value of the variable outside the subshell.
  create_and_use_new_namespace() {
    namespace_number=$(date +%s%N)
    kube::log::status "Creating namespace namespace${namespace_number}"
    kubectl create namespace "namespace${namespace_number}"
    kubectl config set-context "${CONTEXT}" --namespace="namespace${namespace_number}"
  }

  kube_flags=(
    -s "http://127.0.0.1:${API_PORT}"
  )

  # token defined in hack/testdata/auth-tokens.csv
  kube_flags_with_token=(
    -s "https://127.0.0.1:${SECURE_API_PORT}" --token=admin-token --insecure-skip-tls-verify=true
  )

  if [[ -z "${ALLOW_SKEW:-}" ]]; then
    kube_flags+=("--match-server-version")
    kube_flags_with_token+=("--match-server-version")
  fi
  if kube::test::if_supports_resource "${nodes}" ; then
    [ "$(kubectl get nodes -o go-template='{{ .apiVersion }}' "${kube_flags[@]}")" == "v1" ]
  fi

  id_field=".metadata.name"
  labels_field=".metadata.labels"
  annotations_field=".metadata.annotations"
  service_selector_field=".spec.selector"
  rc_replicas_field=".spec.replicas"
  rc_status_replicas_field=".status.replicas"
  rc_container_image_field=".spec.template.spec.containers"
  rs_replicas_field=".spec.replicas"
  port_field="(index .spec.ports 0).port"
  port_name="(index .spec.ports 0).name"
  second_port_field="(index .spec.ports 1).port"
  second_port_name="(index .spec.ports 1).name"
  image_field="(index .spec.containers 0).image"
  pod_container_name_field="(index .spec.containers 0).name"
  container_name_field="(index .spec.template.spec.containers 0).name"
  hpa_min_field=".spec.minReplicas"
  hpa_max_field=".spec.maxReplicas"
  hpa_cpu_field=".spec.targetCPUUtilizationPercentage"
  statefulset_replicas_field=".spec.replicas"
  statefulset_observed_generation=".status.observedGeneration"
  job_parallelism_field=".spec.parallelism"
  deployment_replicas=".spec.replicas"
  secret_data=".data"
  secret_type=".type"
  change_cause_annotation='.*kubernetes.io/change-cause.*'
  pdb_min_available=".spec.minAvailable"
  pdb_max_unavailable=".spec.maxUnavailable"
  template_generation_field=".spec.templateGeneration"
  container_len="(len .spec.template.spec.containers)"
  image_field0="(index .spec.template.spec.containers 0).image"
  image_field1="(index .spec.template.spec.containers 1).image"

  # Make sure "default" namespace exists.
  if kube::test::if_supports_resource "${namespaces}" ; then
    output_message=$(kubectl get "${kube_flags[@]}" namespaces)
    if [[ ! $(echo "${output_message}" | grep "default") ]]; then
      # Create default namespace
      kubectl create "${kube_flags[@]}" ns default
    fi
  fi

  # Make sure "kubernetes" service exists.
  if kube::test::if_supports_resource "${services}" ; then
    # Attempt to create the kubernetes service, tolerating failure (since it might already exist)
    kubectl create "${kube_flags[@]}" -f hack/testdata/kubernetes-service.yaml || true
    # Require the service to exist (either we created it or the API server did)
    kubectl get "${kube_flags[@]}" -f hack/testdata/kubernetes-service.yaml
  fi

  #########################
  # Kubectl version #
  #########################

  record_command run_kubectl_version_tests

  #######################
  # kubectl config set #
  #######################

  record_command run_kubectl_config_set_tests

  #######################
  # kubectl local proxy #
  #######################

  record_command run_kubectl_local_proxy_tests

  #########################
  # RESTMapper evaluation #
  #########################

  record_command run_RESTMapper_evaluation_tests

  ################
  # Cluster Role #
  ################

  if kube::test::if_supports_resource "${clusterroles}" ; then
    record_command run_clusterroles_tests
  fi

  ########
  # Role #
  ########
  if kube::test::if_supports_resource "${roles}" ; then
      record_command run_role_tests
  fi

  #########################
  # Assert short name     #
  #########################

  record_command run_assert_short_name_tests

  #########################
  # Assert categories     #
  #########################

  ## test if a category is exported during discovery
  if kube::test::if_supports_resource "${pods}" ; then
    record_command run_assert_categories_tests
  fi

  ###########################
  # POD creation / deletion #
  ###########################

  if kube::test::if_supports_resource "${pods}" ; then
    record_command run_pod_tests
  fi

  if kube::test::if_supports_resource "${pods}" ; then
    record_command run_save_config_tests
  fi

  if kube::test::if_supports_resource "${pods}" ; then
    record_command run_kubectl_create_error_tests
  fi

  if kube::test::if_supports_resource "${pods}" ; then
    # TODO: Move apply tests to run on rs instead of pods so that they can be
    # run for federation apiserver as well.
    record_command run_kubectl_apply_tests
    record_command run_kubectl_run_tests
    record_command run_kubectl_create_filter_tests
  fi

  if kube::test::if_supports_resource "${deployments}" ; then
    record_command run_kubectl_apply_deployments_tests
  fi

  ###############
  # Kubectl get #
  ###############

  if kube::test::if_supports_resource "${pods}" ; then
    # TODO: Move get tests to run on rs instead of pods so that they can be
    # run for federation apiserver as well.
    record_command run_kubectl_get_tests
  fi

  ##################
  # Global timeout #
  ##################

  if kube::test::if_supports_resource "${pods}" ; then
    # TODO: Move request timeout tests to run on rs instead of pods so that they
    # can be run for federation apiserver as well.
    record_command run_kubectl_request_timeout_tests
  fi

  #####################################
  # Third Party Resources             #
  #####################################

  # customresourcedefinitions cleanup after themselves.  Run these first, then TPRs
  if kube::test::if_supports_resource "${customresourcedefinitions}" ; then
    record_command run_crd_tests
  fi

  #################
  # Run cmd w img #
  #################

  if kube::test::if_supports_resource "${deployments}" ; then
    record_command run_cmd_with_img_tests
  fi


  #####################################
  # Recursive Resources via directory #
  #####################################

  if kube::test::if_supports_resource "${pods}" ; then
    record_command run_recursive_resources_tests
  fi


  ##############
  # Namespaces #
  ##############
  if kube::test::if_supports_resource "${namespaces}" ; then
    record_command run_namespace_tests
  fi


  ###########
  # Secrets #
  ###########
  if kube::test::if_supports_resource "${namespaces}" ; then
    if kube::test::if_supports_resource "${secrets}" ; then
      record_command run_secrets_test
    fi
  fi


  ######################
  # ConfigMap          #
  ######################

  if kube::test::if_supports_resource "${namespaces}"; then
    if kube::test::if_supports_resource "${configmaps}" ; then
      record_command run_configmap_tests
    fi
  fi

  ####################
  # Client Config    #
  ####################

  record_command run_client_config_tests

  ####################
  # Service Accounts #
  ####################

  if kube::test::if_supports_resource "${namespaces}" && kube::test::if_supports_resource "${serviceaccounts}" ; then
    record_command run_service_accounts_tests
  fi

  #################
  # Pod templates #
  #################

  if kube::test::if_supports_resource "${podtemplates}" ; then
    record_command run_pod_templates_tests
  fi

  ############
  # Services #
  ############

  if kube::test::if_supports_resource "${services}" ; then
    record_command run_service_tests
  fi

  ##################
  # DaemonSets     #
  ##################

  if kube::test::if_supports_resource "${daemonsets}" ; then
    record_command run_daemonset_tests
    if kube::test::if_supports_resource "${controllerrevisions}"; then
      record_command run_daemonset_history_tests
    fi
  fi

  ###########################
  # Replication controllers #
  ###########################

  if kube::test::if_supports_resource "${namespaces}" ; then
    if kube::test::if_supports_resource "${replicationcontrollers}" ; then
      record_command run_rc_tests
    fi
  fi

  ######################
  # Deployments       #
  ######################

  if kube::test::if_supports_resource "${deployments}" ; then
    record_command run_deployment_tests
  fi

  ######################
  # Replica Sets       #
  ######################

  if kube::test::if_supports_resource "${replicasets}" ; then
    record_command run_rs_tests
  fi

  #################
  # Stateful Sets #
  #################

  if kube::test::if_supports_resource "${statefulsets}" ; then
    record_command run_stateful_set_tests
    if kube::test::if_supports_resource "${controllerrevisions}"; then
      record_command run_statefulset_history_tests
    fi
  fi

  ######################
  # Lists              #
  ######################

  if kube::test::if_supports_resource "${services}" ; then
    if kube::test::if_supports_resource "${deployments}" ; then
      record_command run_lists_tests
    fi
  fi


  ######################
  # Multiple Resources #
  ######################
  if kube::test::if_supports_resource "${services}" ; then
    if kube::test::if_supports_resource "${replicationcontrollers}" ; then
      record_command run_multi_resources_tests
    fi
  fi

  ######################
  # Persistent Volumes #
  ######################

  if kube::test::if_supports_resource "${persistentvolumes}" ; then
    record_command run_persistent_volumes_tests
  fi

  ############################
  # Persistent Volume Claims #
  ############################

  if kube::test::if_supports_resource "${persistentvolumeclaims}" ; then
    record_command run_persistent_volume_claims_tests
  fi

  ############################
  # Storage Classes #
  ############################

  if kube::test::if_supports_resource "${storageclass}" ; then
    record_command run_storage_class_tests
  fi

  #########
  # Nodes #
  #########

  if kube::test::if_supports_resource "${nodes}" ; then
    record_command run_nodes_tests
  fi


  ########################
  # authorization.k8s.io #
  ########################

  if kube::test::if_supports_resource "${subjectaccessreviews}" ; then
    record_command run_authorization_tests
  fi

  # kubectl auth can-i
  # kube-apiserver is started with authorization mode AlwaysAllow, so kubectl can-i always returns yes
  if kube::test::if_supports_resource "${subjectaccessreviews}" ; then
    output_message=$(kubectl auth can-i '*' '*' 2>&1 "${kube_flags[@]}")
    kube::test::if_has_string "${output_message}" "yes"

    output_message=$(kubectl auth can-i get pods --subresource=log 2>&1 "${kube_flags[@]}")
    kube::test::if_has_string "${output_message}" "yes"

    output_message=$(kubectl auth can-i get invalid_resource 2>&1 "${kube_flags[@]}")
    kube::test::if_has_string "${output_message}" "the server doesn't have a resource type"

    output_message=$(kubectl auth can-i get /logs/ 2>&1 "${kube_flags[@]}")
    kube::test::if_has_string "${output_message}" "yes"

    output_message=$(! kubectl auth can-i get /logs/ --subresource=log 2>&1 "${kube_flags[@]}")
    kube::test::if_has_string "${output_message}" "subresource can not be used with nonResourceURL"

    output_message=$(kubectl auth can-i list jobs.batch/bar -n foo --quiet 2>&1 "${kube_flags[@]}")
    kube::test::if_empty_string "${output_message}"
  fi

  # kubectl auth reconcile
  if kube::test::if_supports_resource "${clusterroles}" ; then
    kubectl auth reconcile "${kube_flags[@]}" -f test/fixtures/pkg/kubectl/cmd/auth/rbac-resource-plus.yaml
    kube::test::get_object_assert 'rolebindings -n some-other-random -l test-cmd=auth' "{{range.items}}{{$id_field}}:{{end}}" 'testing-RB:'
    kube::test::get_object_assert 'roles -n some-other-random -l test-cmd=auth' "{{range.items}}{{$id_field}}:{{end}}" 'testing-R:'
    kube::test::get_object_assert 'clusterrolebindings -l test-cmd=auth' "{{range.items}}{{$id_field}}:{{end}}" 'testing-CRB:'
    kube::test::get_object_assert 'clusterroles -l test-cmd=auth' "{{range.items}}{{$id_field}}:{{end}}" 'testing-CR:'

    kubectl delete "${kube_flags[@]}" rolebindings,role,clusterroles,clusterrolebindings -n some-other-random -l test-cmd=auth
  fi

  #####################
  # Retrieve multiple #
  #####################

  if kube::test::if_supports_resource "${nodes}" ; then
    if kube::test::if_supports_resource "${services}" ; then
      record_command run_retrieve_multiple_tests
    fi
  fi


  #####################
  # Resource aliasing #
  #####################

  if kube::test::if_supports_resource "${services}" ; then
    if kube::test::if_supports_resource "${replicationcontrollers}" ; then
      record_command run_resource_aliasing_tests
    fi
  fi

  ###########
  # Explain #
  ###########

  if kube::test::if_supports_resource "${pods}" ; then
    record_command run_kubectl_explain_tests
  fi


  ###########
  # Swagger #
  ###########

  record_command run_swagger_tests

  #####################
  # Kubectl --sort-by #
  #####################

  if kube::test::if_supports_resource "${pods}" ; then
    record_command run_kubectl_sort_by_tests
  fi

  ############################
  # Kubectl --all-namespaces #
  ############################

  if kube::test::if_supports_resource "${pods}" ; then
    record_command run_kubectl_all_namespace_tests
  fi

  ################
  # Certificates #
  ################

  if kube::test::if_supports_resource "${csr}" ; then
    record_command run_certificates_tests
  fi

  ###########
  # Plugins #
  ###########

  record_command run_plugins_tests

  #################
  # Impersonation #
  #################
  record_command run_impersonation_tests

  kube::test::clear_all

  if [ "$foundError" == "True" ]; then
    echo "TEST FAILED"
    exit 1
  fi
}

run_initializer_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing --include-uninitialized"

  ### Create a deployment
  kubectl create --request-timeout=1 -f hack/testdata/initializer-deployments.yaml 2>&1 "${kube_flags[@]}" || true

  ### Test kubectl get --include-uninitialized
  # Command
  output_message=$(kubectl get deployments 2>&1 "${kube_flags[@]}")
  # Post-condition: The text "No resources found" should be part of the output
  kube::test::if_has_string "${output_message}" 'No resources found'
  # Command
  output_message=$(kubectl get deployments --include-uninitialized=false 2>&1 "${kube_flags[@]}")
  # Post-condition: The text "No resources found" should be part of the output
  kube::test::if_has_string "${output_message}" 'No resources found'
  # Command
  output_message=$(kubectl get deployments --include-uninitialized 2>&1 "${kube_flags[@]}")
  # Post-condition: I assume "web" is the deployment name
  kube::test::if_has_string "${output_message}" 'web'
  # Command
  output_message=$(kubectl get deployments web 2>&1 "${kube_flags[@]}")
  # Post-condition: I assume "web" is the deployment name
  kube::test::if_has_string "${output_message}" 'web'
  # Command
  output_message=$(kubectl get deployments --show-all 2>&1 "${kube_flags[@]}")
  # Post-condition: The text "No resources found" should be part of the output
  kube::test::if_has_string "${output_message}" 'No resources found'

  ### Test kubectl describe --include-uninitialized
  # Command
  output_message=$(kubectl describe deployments 2>&1 "${kube_flags[@]}")
  # Post-condition: The text "run=web" should be part of the output
  kube::test::if_has_string "${output_message}" 'run=web'
  # Command
  output_message=$(kubectl describe deployments --include-uninitialized 2>&1 "${kube_flags[@]}")
  # Post-condition: The text "run=web" should be part of the output
  kube::test::if_has_string "${output_message}" 'run=web'
  # Command
  output_message=$(kubectl describe deployments --include-uninitialized=false 2>&1 "${kube_flags[@]}")
  # Post-condition: The output should be empty
  kube::test::if_empty_string "${output_message}"
  # Command
  output_message=$(kubectl describe deployments web --include-uninitialized 2>&1 "${kube_flags[@]}")
  # Post-condition: The text "run=web" should be part of the output
  kube::test::if_has_string "${output_message}" 'run=web'
  # Command
  output_message=$(kubectl describe deployments web --include-uninitialized=false 2>&1 "${kube_flags[@]}")
  # Post-condition: The text "run=web" should be part of the output
  kube::test::if_has_string "${output_message}" 'run=web'

  ### Test kubectl label --include-uninitialized
  # Command
  output_message=$(kubectl label deployments labelkey1=labelvalue1 --all 2>&1 "${kube_flags[@]}")
  # Post-condition: web is labelled
  kube::test::if_has_string "${output_message}" 'deployment "web" labeled'
  kube::test::get_object_assert 'deployments web' "{{${labels_field}.labelkey1}}" 'labelvalue1'
  # Command
  output_message=$(kubectl label deployments labelkey2=labelvalue2 --all --include-uninitialized=false 2>&1 "${kube_flags[@]}")
  # Post-condition: The output should be empty
  kube::test::if_empty_string "${output_message}"
  # Command
  output_message=$(kubectl label deployments labelkey3=labelvalue3 -l run=web 2>&1 "${kube_flags[@]}")
  # Post-condition: The output should be empty
  kube::test::if_empty_string "${output_message}"
  # Command
  output_message=$(kubectl label deployments labelkey4=labelvalue4 -l run=web --include-uninitialized 2>&1 "${kube_flags[@]}")
  # Post-condition: web is labelled
  kube::test::if_has_string "${output_message}" 'deployment "web" labeled'
  kube::test::get_object_assert 'deployments web' "{{${labels_field}.labelkey4}}" 'labelvalue4'
  # Command
  output_message=$(kubectl label deployments labelkey5=labelvalue5 -l run=web --all 2>&1 "${kube_flags[@]}")
  # Post-condition: The output should be empty
  kube::test::if_empty_string "${output_message}"
  # Command
  output_message=$(kubectl label deployments labelkey6=labelvalue6 -l run=web --all --include-uninitialized 2>&1 "${kube_flags[@]}")
  # Post-condition: web is labelled
  kube::test::if_has_string "${output_message}" 'deployment "web" labeled'
  kube::test::get_object_assert 'deployments web' "{{${labels_field}.labelkey6}}" 'labelvalue6'
  # Command
  output_message=$(kubectl label deployments web labelkey7=labelvalue7 2>&1 "${kube_flags[@]}")
  # Post-condition: web is labelled
  kube::test::if_has_string "${output_message}" 'deployment "web" labeled'
  kube::test::get_object_assert 'deployments web' "{{${labels_field}.labelkey7}}" 'labelvalue7'
  # Found All Labels
  kube::test::get_object_assert 'deployments web' "{{${labels_field}}}" 'map[labelkey1:labelvalue1 labelkey4:labelvalue4 labelkey6:labelvalue6 labelkey7:labelvalue7 run:web]'

  ### Test kubectl annotate --include-uninitialized
  # Command
  output_message=$(kubectl annotate deployments annotatekey1=annotatevalue1 --all 2>&1 "${kube_flags[@]}")
  # Post-condition: DEPLOYMENT has annotation
  kube::test::if_has_string "${output_message}" 'deployment "web" annotated'
  kube::test::get_object_assert 'deployments web' "{{${annotations_field}.annotatekey1}}" 'annotatevalue1'
  # Command
  output_message=$(kubectl annotate deployments annotatekey2=annotatevalue2 --all --include-uninitialized=false 2>&1 "${kube_flags[@]}")
  # Post-condition: The output should be empty
  kube::test::if_empty_string "${output_message}"
  # Command
  output_message=$(kubectl annotate deployments annotatekey3=annotatevalue3 -l run=web 2>&1 "${kube_flags[@]}")
  # Post-condition: The output should be empty
  kube::test::if_empty_string "${output_message}"
  # Command
  output_message=$(kubectl annotate deployments annotatekey4=annotatevalue4 -l run=web --include-uninitialized 2>&1 "${kube_flags[@]}")
  # Post-condition: DEPLOYMENT has annotation
  kube::test::if_has_string "${output_message}" 'deployment "web" annotated'
  kube::test::get_object_assert 'deployments web' "{{${annotations_field}.annotatekey4}}" 'annotatevalue4'
  # Command
  output_message=$(kubectl annotate deployments annotatekey5=annotatevalue5 -l run=web --all 2>&1 "${kube_flags[@]}")
  # Post-condition: The output should be empty
  kube::test::if_empty_string "${output_message}"
  # Command
  output_message=$(kubectl annotate deployments annotatekey6=annotatevalue6 -l run=web --all --include-uninitialized 2>&1 "${kube_flags[@]}")
  # Post-condition: DEPLOYMENT has annotation
  kube::test::if_has_string "${output_message}" 'deployment "web" annotated'
  kube::test::get_object_assert 'deployments web' "{{${annotations_field}.annotatekey6}}" 'annotatevalue6'
  # Command
  output_message=$(kubectl annotate deployments web annotatekey7=annotatevalue7 2>&1 "${kube_flags[@]}")
  # Post-condition: web DEPLOYMENT has annotation
  kube::test::if_has_string "${output_message}" 'deployment "web" annotated'
  kube::test::get_object_assert 'deployments web' "{{${annotations_field}.annotatekey7}}" 'annotatevalue7'

  ### Test kubectl edit --include-uninitialized
  [ "$(EDITOR=cat kubectl edit deployments 2>&1 "${kube_flags[@]}" | grep 'edit cancelled, no objects found')" ]
  [ "$(EDITOR=cat kubectl edit deployments --include-uninitialized 2>&1 "${kube_flags[@]}" | grep 'Edit cancelled, no changes made.')" ]

  ### Test kubectl set image --include-uninitialized
  # Command
  output_message=$(kubectl set image deployments *=nginx:1.11 --all 2>&1 "${kube_flags[@]}")
  # Post-condition: The text "image updated" should be part of the output
  kube::test::if_has_string "${output_message}" 'image updated'
  # Command
  output_message=$(kubectl set image deployments *=nginx:1.11 --all --include-uninitialized=false 2>&1 "${kube_flags[@]}")
  # Post-condition: The output should be empty
  kube::test::if_empty_string "${output_message}"
  # Command
  output_message=$(kubectl set image deployments *=nginx:1.11 -l run=web 2>&1 "${kube_flags[@]}")
  # Post-condition: The output should be empty
  kube::test::if_empty_string "${output_message}"
  # Command
  output_message=$(kubectl set image deployments *=nginx:1.12 -l run=web --include-uninitialized 2>&1 "${kube_flags[@]}")
  # Post-condition: The text "image updated" should be part of the output
  kube::test::if_has_string "${output_message}" 'image updated'
  # Command
  output_message=$(kubectl set image deployments *=nginx:1.13 -l run=web --include-uninitialized --all 2>&1 "${kube_flags[@]}")
  # Post-condition: The text "image updated" should be part of the output
  kube::test::if_has_string "${output_message}" 'image updated'

  ### Test kubectl set resources --include-uninitialized
  # Command
  output_message=$(kubectl set resources deployments --limits=cpu=200m,memory=512Mi --requests=cpu=100m,memory=256Mi --all 2>&1 "${kube_flags[@]}")
  # Post-condition: The text "resource requirements updated" should be part of the output
  kube::test::if_has_string "${output_message}" 'resource requirements updated'
  # Command
  output_message=$(kubectl set resources deployments --limits=cpu=200m,memory=512Mi --requests=cpu=100m,memory=256Mi --all --include-uninitialized=false 2>&1 "${kube_flags[@]}")
  # Post-condition: The output should be empty
  kube::test::if_empty_string "${output_message}"
  # Command
  output_message=$(kubectl set resources deployments --limits=cpu=200m,memory=512Mi --requests=cpu=100m,memory=256Mi -l run=web 2>&1 "${kube_flags[@]}")
  # Post-condition: The output should be empty
  kube::test::if_empty_string "${output_message}"
  # Command
  output_message=$(kubectl set resources deployments --limits=cpu=200m,memory=512Mi --requests=cpu=200m,memory=256Mi -l run=web --include-uninitialized 2>&1 "${kube_flags[@]}")
  # Post-condition: The text "resource requirements updated" should be part of the output
  kube::test::if_has_string "${output_message}" 'resource requirements updated'
  # Command
  output_message=$(kubectl set resources deployments --limits=cpu=200m,memory=512Mi --requests=cpu=100m,memory=512Mi -l run=web --include-uninitialized --all 2>&1 "${kube_flags[@]}")
  # Post-condition: The text "resource requirements updated" should be part of the output
  kube::test::if_has_string "${output_message}" 'resource requirements updated'

  ### Test kubectl set selector --include-uninitialized
  # Create a service with initializer
  kubectl create --request-timeout=1 -f hack/testdata/initializer-redis-master-service.yaml 2>&1 "${kube_flags[@]}" || true
  # Command
  output_message=$(kubectl set selector services role=padawan --all 2>&1 "${kube_flags[@]}")
  # Post-condition: The text "selector updated" should be part of the output
  kube::test::if_has_string "${output_message}" 'selector updated'
  # Command
  output_message=$(kubectl set selector services role=padawan --all --include-uninitialized=false 2>&1 "${kube_flags[@]}")
  # Post-condition: The output should be empty
  kube::test::if_empty_string "${output_message}"

  ### Test kubectl set subject --include-uninitialized
  # Create a create clusterrolebinding with initializer
  kubectl create --request-timeout=1 -f hack/testdata/initializer-clusterrolebinding.yaml 2>&1 "${kube_flags[@]}" || true
  kube::test::get_object_assert clusterrolebinding/super-admin "{{range.subjects}}{{.name}}:{{end}}" 'super-admin:'
  # Command
  output_message=$(kubectl set subject clusterrolebinding --user=foo --all 2>&1 "${kube_flags[@]}")
  # Post-condition: The text "subjects updated" should be part of the output
  kube::test::if_has_string "${output_message}" 'subjects updated'
  # Command
  output_message=$(kubectl set subject clusterrolebinding --user=foo --all --include-uninitialized=false 2>&1 "${kube_flags[@]}")
  # Post-condition: The output should be empty
  kube::test::if_empty_string "${output_message}"
  # Command
  output_message=$(kubectl set subject clusterrolebinding --user=foo -l clusterrolebinding=super 2>&1 "${kube_flags[@]}")
  # Post-condition: The output should be empty
  kube::test::if_empty_string "${output_message}"
  # Command
  output_message=$(kubectl set subject clusterrolebinding --user=foo -l clusterrolebinding=super --include-uninitialized 2>&1 "${kube_flags[@]}")
  # Post-condition: The text "subjects updated" should be part of the output
  kube::test::if_has_string "${output_message}" 'subjects updated'
  # Command
  output_message=$(kubectl set subject clusterrolebinding --user=foo -l clusterrolebinding=super --include-uninitialized --all 2>&1 "${kube_flags[@]}")
  # Post-condition: The text "subjects updated" should be part of the output
  kube::test::if_has_string "${output_message}" 'subjects updated'

  ### Test kubectl set serviceaccount --include-uninitialized
  # Command
  output_message=$(kubectl set serviceaccount deployment serviceaccount1 --all 2>&1 "${kube_flags[@]}")
  # Post-condition: The text "serviceaccount updated" should be part of the output
  kube::test::if_has_string "${output_message}" 'serviceaccount updated'
  # Command
  output_message=$(kubectl set serviceaccount deployment serviceaccount1 --all --include-uninitialized=false 2>&1 "${kube_flags[@]}")
  # Post-condition: The output should be empty
  kube::test::if_empty_string "${output_message}"

  ### Test kubectl delete --include-uninitialized
  kube::test::get_object_assert clusterrolebinding/super-admin "{{range.subjects}}{{.name}}:{{end}}" 'super-admin:'
  # Command
  output_message=$(kubectl delete clusterrolebinding --all --include-uninitialized=false 2>&1 "${kube_flags[@]}")
  # Post-condition: The text "No resources found" should be part of the output
  kube::test::if_has_string "${output_message}" 'No resources found'
  # Command
  output_message=$(kubectl delete clusterrolebinding --all 2>&1 "${kube_flags[@]}")
  # Post-condition: The text "deleted" should be part of the output
  kube::test::if_has_string "${output_message}" 'deleted'
  kube::test::get_object_assert clusterrolebinding/super-admin "{{range.items}}{{$id_field}}:{{end}}" ''

  ### Test kubectl apply --include-uninitialized
  # Pre-Condition: no POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # apply pod a
  kubectl apply --prune --request-timeout=20 --include-uninitialized=false --all -f hack/testdata/prune/a.yaml "${kube_flags[@]}" 2>&1
  # check right pod exists
  kube::test::get_object_assert pods/a "{{${id_field}}}" 'a'
  # Post-condition: Other uninitialized resources should not be pruned
  kube::test::get_object_assert deployments "{{range.items}}{{$id_field}}:{{end}}" 'web'
  kube::test::get_object_assert services/redis-master "{{range.items}}{{$id_field}}:{{end}}" 'redis-master'
  # cleanup
  kubectl delete pod a
  # apply pod a and prune uninitialized deployments web
  kubectl apply --prune --request-timeout=20 --all -f hack/testdata/prune/a.yaml "${kube_flags[@]}" 2>&1
  # check right pod exists
  kube::test::get_object_assert pods/a "{{${id_field}}}" 'a'
  # Post-condition: Other uninitialized resources should not be pruned
  kube::test::get_object_assert deployments/web "{{range.items}}{{$id_field}}:{{end}}" 'web'
  kube::test::get_object_assert services/redis-master "{{range.items}}{{$id_field}}:{{end}}" 'redis-master'
  # cleanup
  kubectl delete pod a
  # apply pod a and prune uninitialized deployments web
  kubectl apply --prune --request-timeout=20 --include-uninitialized --all -f hack/testdata/prune/a.yaml "${kube_flags[@]}" 2>&1
  # check right pod exists
  kube::test::get_object_assert pods/a "{{${id_field}}}" 'a'
  # Post-condition: Other uninitialized resources should not be pruned
  kube::test::get_object_assert deployments/web "{{range.items}}{{$id_field}}:{{end}}" 'web'
  kube::test::get_object_assert services/redis-master "{{range.items}}{{$id_field}}:{{end}}" 'redis-master'
  # cleanup
  kubectl delete pod a
  kubectl delete --request-timeout=1 deploy web
  kubectl delete --request-timeout=1 service redis-master

  set +o nounset
  set +o errexit
}
