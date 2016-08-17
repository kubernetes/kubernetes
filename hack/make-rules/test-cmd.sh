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

# This command checks that the built commands can function together for
# simple scenarios.  It does not require Docker.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
source "${KUBE_ROOT}/hack/lib/init.sh"
source "${KUBE_ROOT}/hack/lib/test.sh"

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

kube::util::trap_add cleanup EXIT SIGINT
kube::util::ensure-temp-dir

BINS=(
	cmd/kubectl
	cmd/kube-apiserver
	cmd/kube-controller-manager
)
make -C "${KUBE_ROOT}" WHAT="${BINS[*]}"

kube::etcd::start

ETCD_HOST=${ETCD_HOST:-127.0.0.1}
ETCD_PORT=${ETCD_PORT:-2379}
API_PORT=${API_PORT:-8080}
API_HOST=${API_HOST:-127.0.0.1}
KUBELET_PORT=${KUBELET_PORT:-10250}
KUBELET_HEALTHZ_PORT=${KUBELET_HEALTHZ_PORT:-10248}
CTLRMGR_PORT=${CTLRMGR_PORT:-10252}
PROXY_HOST=127.0.0.1 # kubectl only serves on localhost.

IMAGE_NGINX="gcr.io/google-containers/nginx:1.7.9"
IMAGE_DEPLOYMENT_R1="gcr.io/google-containers/nginx:test-cmd"  # deployment-revision1.yaml
IMAGE_DEPLOYMENT_R2="$IMAGE_NGINX"  # deployment-revision2.yaml
IMAGE_PERL="gcr.io/google-containers/perl"

# ensure ~/.kube/config isn't loaded by tests
HOME="${KUBE_TEMP}"

# Find a standard sed instance for use with edit scripts
SED=sed
if which gsed &>/dev/null; then
  SED=gsed
fi
if ! ($SED --version 2>&1 | grep -q GNU); then
  echo "!!! GNU sed is required.  If on OS X, use 'brew install gnu-sed'."
  exit 1
fi

# Check kubectl
kube::log::status "Running kubectl with no options"
"${KUBE_OUTPUT_HOSTBIN}/kubectl"

# Only run kubelet on platforms it supports
if [[ "$(go env GOHOSTOS)" == "linux" ]]; then

BINS=(
	cmd/kubelet
)
make -C "${KUBE_ROOT}" WHAT="${BINS[*]}"

kube::log::status "Starting kubelet in masterless mode"
"${KUBE_OUTPUT_HOSTBIN}/kubelet" \
  --really-crash-for-testing=true \
  --root-dir=/tmp/kubelet.$$ \
  --cert-dir="${TMPDIR:-/tmp/}" \
  --docker-endpoint="fake://" \
  --hostname-override="127.0.0.1" \
  --address="127.0.0.1" \
  --port="$KUBELET_PORT" \
  --healthz-port="${KUBELET_HEALTHZ_PORT}" 1>&2 &
KUBELET_PID=$!
kube::util::wait_for_url "http://127.0.0.1:${KUBELET_HEALTHZ_PORT}/healthz" "kubelet(masterless)"
kill ${KUBELET_PID} 1>&2 2>/dev/null

kube::log::status "Starting kubelet in masterful mode"
"${KUBE_OUTPUT_HOSTBIN}/kubelet" \
  --really-crash-for-testing=true \
  --root-dir=/tmp/kubelet.$$ \
  --cert-dir="${TMPDIR:-/tmp/}" \
  --docker-endpoint="fake://" \
  --hostname-override="127.0.0.1" \
  --address="127.0.0.1" \
  --api-servers="${API_HOST}:${API_PORT}" \
  --port="$KUBELET_PORT" \
  --healthz-port="${KUBELET_HEALTHZ_PORT}" 1>&2 &
KUBELET_PID=$!

kube::util::wait_for_url "http://127.0.0.1:${KUBELET_HEALTHZ_PORT}/healthz" "kubelet"

fi

# Start kube-apiserver
kube::log::status "Starting kube-apiserver"

# Admission Controllers to invoke prior to persisting objects in cluster
ADMISSION_CONTROL="NamespaceLifecycle,LimitRanger,ResourceQuota"

"${KUBE_OUTPUT_HOSTBIN}/kube-apiserver" \
  --address="127.0.0.1" \
  --public-address-override="127.0.0.1" \
  --port="${API_PORT}" \
  --admission-control="${ADMISSION_CONTROL}" \
  --etcd-servers="http://${ETCD_HOST}:${ETCD_PORT}" \
  --public-address-override="127.0.0.1" \
  --kubelet-port=${KUBELET_PORT} \
  --runtime-config=api/v1 \
  --storage-media-type="${KUBE_TEST_API_STORAGE_TYPE-}" \
  --cert-dir="${TMPDIR:-/tmp/}" \
  --service-cluster-ip-range="10.0.0.0/24" 1>&2 &
APISERVER_PID=$!

kube::util::wait_for_url "http://127.0.0.1:${API_PORT}/healthz" "apiserver"

# Start controller manager
kube::log::status "Starting controller-manager"
"${KUBE_OUTPUT_HOSTBIN}/kube-controller-manager" \
  --port="${CTLRMGR_PORT}" \
  --kube-api-content-type="${KUBE_TEST_API_TYPE-}" \
  --master="127.0.0.1:${API_PORT}" 1>&2 &
CTLRMGR_PID=$!

kube::util::wait_for_url "http://127.0.0.1:${CTLRMGR_PORT}/healthz" "controller-manager"

if [[ "$(go env GOHOSTOS)" == "linux" ]]; then
  kube::util::wait_for_url "http://127.0.0.1:${API_PORT}/api/v1/nodes/127.0.0.1" "apiserver(nodes)"
else
  # create a fake node
  kubectl create -f - -s "http://127.0.0.1:${API_PORT}" << __EOF__
{
  "kind": "Node",
  "apiVersion": "v1",
  "metadata": {
    "name": "127.0.0.1"
  },
  "status": {
    "capacity": {
      "memory": "1Gi"
    }
  }
}
__EOF__
fi

# Expose kubectl directly for readability
PATH="${KUBE_OUTPUT_HOSTBIN}":$PATH

kube::log::status "Checking kubectl version"
kubectl version

# TODO: we need to note down the current default namespace and set back to this
# namespace after the tests are done.
kubectl config view
CONTEXT="test"
kubectl config set-context "${CONTEXT}"
kubectl config use-context "${CONTEXT}"

i=0
create_and_use_new_namespace() {
  i=$(($i+1))
  kubectl create namespace "namespace${i}"
  kubectl config set-context "${CONTEXT}" --namespace="namespace${i}"
}

runTests() {
  version="$1"
  echo "Testing api version: $1"
  if [[ -z "${version}" ]]; then
    kube_flags=(
      -s "http://127.0.0.1:${API_PORT}"
      --match-server-version
    )
    [ "$(kubectl get nodes -o go-template='{{ .apiVersion }}' "${kube_flags[@]}")" == "v1" ]
  else
    kube_flags=(
      -s "http://127.0.0.1:${API_PORT}"
      --match-server-version
    )
    [ "$(kubectl get nodes -o go-template='{{ .apiVersion }}' "${kube_flags[@]}")" == "${version}" ]
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
  hpa_min_field=".spec.minReplicas"
  hpa_max_field=".spec.maxReplicas"
  hpa_cpu_field=".spec.targetCPUUtilizationPercentage"
  job_parallelism_field=".spec.parallelism"
  deployment_replicas=".spec.replicas"
  secret_data=".data"
  secret_type=".type"
  deployment_image_field="(index .spec.template.spec.containers 0).image"
  deployment_second_image_field="(index .spec.template.spec.containers 1).image"
  change_cause_annotation='.*kubernetes.io/change-cause.*'

  # Passing no arguments to create is an error
  ! kubectl create

  #######################
  # kubectl config set #
  #######################

  kube::log::status "Testing kubectl(${version}:config set)"

  kubectl config set-cluster test-cluster --server="https://does-not-work"

  # Get the api cert and add a comment to avoid flag parsing problems
  cert_data=$(echo "#Comment" && cat "${TMPDIR:-/tmp}/apiserver.crt")

  kubectl config set clusters.test-cluster.certificate-authority-data "$cert_data" --set-raw-bytes
  r_writen=$(kubectl config view --raw -o jsonpath='{.clusters[?(@.name == "test-cluster")].cluster.certificate-authority-data}')

  encoded=$(echo -n "$cert_data" | base64)
  kubectl config set clusters.test-cluster.certificate-authority-data "$encoded"
  e_writen=$(kubectl config view --raw -o jsonpath='{.clusters[?(@.name == "test-cluster")].cluster.certificate-authority-data}')

  test "$e_writen" == "$r_writen"

  #######################
  # kubectl local proxy #
  #######################

  # Make sure the UI can be proxied
  start-proxy
  check-curl-proxy-code /ui 301
  check-curl-proxy-code /metrics 200
  check-curl-proxy-code /api/ui 404
  if [[ -n "${version}" ]]; then
    check-curl-proxy-code /api/${version}/namespaces 200
  fi
  check-curl-proxy-code /static/ 200
  stop-proxy

  # Make sure the in-development api is accessible by default
  start-proxy
  check-curl-proxy-code /apis 200
  check-curl-proxy-code /apis/extensions/ 200
  stop-proxy

  # Custom paths let you see everything.
  start-proxy /custom
  check-curl-proxy-code /custom/ui 301
  check-curl-proxy-code /custom/metrics 200
  if [[ -n "${version}" ]]; then
    check-curl-proxy-code /custom/api/${version}/namespaces 200
  fi
  stop-proxy

  #########################
  # RESTMapper evaluation #
  #########################

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

  ###########################
  # POD creation / deletion #
  ###########################

  kube::log::status "Testing kubectl(${version}:pods)"

  ### Create POD valid-pod from JSON
  # Pre-condition: no POD exists
  create_and_use_new_namespace
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create "${kube_flags[@]}" -f docs/admin/limitrange/valid-pod.yaml
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
  kube::test::describe_object_assert pods 'valid-pod' "Name:" "Image:" "Node:" "Labels:" "Status:" "Controllers"
  # Describe command should print events information by default
  kube::test::describe_object_events_assert pods 'valid-pod'
  # Describe command should not print events information when show-events=false
  kube::test::describe_object_events_assert pods 'valid-pod' false
  # Describe command should print events information when show-events=true
  kube::test::describe_object_events_assert pods 'valid-pod' true
  # Describe command (resource only) should print detailed information
  kube::test::describe_resource_assert pods "Name:" "Image:" "Node:" "Labels:" "Status:" "Controllers"

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
  kubectl delete pod valid-pod "${kube_flags[@]}" --grace-period=0
  # Post-condition: valid-pod POD doesn't exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''

  ### Delete POD valid-pod by id with --now
  # Pre-condition: valid-pod POD exists
  kubectl create "${kube_flags[@]}" -f docs/admin/limitrange/valid-pod.yaml
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'
  # Command
  kubectl delete pod valid-pod "${kube_flags[@]}" --now
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
  kubectl delete -f docs/admin/limitrange/valid-pod.yaml "${kube_flags[@]}" --grace-period=0
  # Post-condition: valid-pod POD doesn't exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''

  ### Create POD valid-pod from JSON
  # Pre-condition: no POD exists
  create_and_use_new_namespace
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f docs/admin/limitrange/valid-pod.yaml "${kube_flags[@]}"
  # Post-condition: valid-pod POD is created
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'

  ### Delete POD valid-pod with label
  # Pre-condition: valid-pod POD exists
  kube::test::get_object_assert "pods -l'name in (valid-pod)'" '{{range.items}}{{$id_field}}:{{end}}' 'valid-pod:'
  # Command
  kubectl delete pods -l'name in (valid-pod)' "${kube_flags[@]}" --grace-period=0
  # Post-condition: valid-pod POD doesn't exist
  kube::test::get_object_assert "pods -l'name in (valid-pod)'" '{{range.items}}{{$id_field}}:{{end}}' ''

  ### Create POD valid-pod from YAML
  # Pre-condition: no POD exists
  create_and_use_new_namespace
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f docs/admin/limitrange/valid-pod.yaml "${kube_flags[@]}"
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
  kubectl delete --all pods "${kube_flags[@]}" --grace-period=0 # --all remove all the pods
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
  kubectl delete namespace test-kubectl-describe-pod

  ### Create two PODs
  # Pre-condition: no POD exists
  create_and_use_new_namespace
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f docs/admin/limitrange/valid-pod.yaml "${kube_flags[@]}"
  kubectl create -f examples/storage/redis/redis-proxy.yaml "${kube_flags[@]}"
  # Post-condition: valid-pod and redis-proxy PODs are created
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'redis-proxy:valid-pod:'

  ### Delete multiple PODs at once
  # Pre-condition: valid-pod and redis-proxy PODs exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'redis-proxy:valid-pod:'
  # Command
  kubectl delete pods valid-pod redis-proxy "${kube_flags[@]}" --grace-period=0 # delete multiple pods at once
  # Post-condition: no POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''

  ### Create valid-pod POD
  # Pre-condition: no POD exists
  create_and_use_new_namespace
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f docs/admin/limitrange/valid-pod.yaml "${kube_flags[@]}"
  # Post-condition: valid-pod POD is created
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'

  ### Label the valid-pod POD
  # Pre-condition: valid-pod is not labelled
  kube::test::get_object_assert 'pod valid-pod' "{{range$labels_field}}{{.}}:{{end}}" 'valid-pod:'
  # Command
  kubectl label pods valid-pod new-name=new-valid-pod "${kube_flags[@]}"
  # Post-condition: valid-pod is labelled
  kube::test::get_object_assert 'pod valid-pod' "{{range$labels_field}}{{.}}:{{end}}" 'valid-pod:new-valid-pod:'

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
  kubectl delete pods -lnew-name=new-valid-pod --grace-period=0 "${kube_flags[@]}"
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

  ### Create valid-pod POD
  # Pre-condition: no POD exists
  create_and_use_new_namespace
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f docs/admin/limitrange/valid-pod.yaml "${kube_flags[@]}"
  # Post-condition: valid-pod POD is created
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'

  ## Patch can modify a local object
  kubectl patch --local -f pkg/api/validation/testdata/v1/validPod.yaml --patch='{"spec": {"restartPolicy":"Never"}}' -o jsonpath='{.spec.restartPolicy}' | grep -q "Never"

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
  kubectl patch "${kube_flags[@]}" -f docs/admin/limitrange/valid-pod.yaml -p='{"spec":{"containers":[{"name": "kubernetes-serve-hostname", "image": "gcr.io/google_containers/pause-amd64:3.0"}]}}'
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
  #cleaning
  rm /tmp/tmp-valid-pod.json

  ## replace of a cluster scoped resource can succeed
  # Pre-condition: a node exists
  kubectl create -f - "${kube_flags[@]}" << __EOF__
{
  "kind": "Node",
  "apiVersion": "v1",
  "metadata": {
    "name": "node-${version}-test"
  }
}
__EOF__
  kubectl replace -f - "${kube_flags[@]}" << __EOF__
{
  "kind": "Node",
  "apiVersion": "v1",
  "metadata": {
    "name": "node-${version}-test",
    "annotations": {"a":"b"}
  }
}
__EOF__
  # Post-condition: the node command succeeds
  kube::test::get_object_assert "node node-${version}-test" "{{.metadata.annotations.a}}" 'b'
  kubectl delete node node-${version}-test "${kube_flags[@]}"

  ## kubectl edit can update the image field of a POD. tmp-editor.sh is a fake editor
  echo -e "#!/bin/bash\n$SED -i \"s/nginx/gcr.io\/google_containers\/serve_hostname/g\" \$1" > /tmp/tmp-editor.sh
  chmod +x /tmp/tmp-editor.sh
  # Pre-condition: valid-pod POD has image nginx
  kube::test::get_object_assert pods "{{range.items}}{{$image_field}}:{{end}}" 'nginx:'
  EDITOR=/tmp/tmp-editor.sh kubectl edit "${kube_flags[@]}" pods/valid-pod
  # Post-condition: valid-pod POD has image gcr.io/google_containers/serve_hostname
  kube::test::get_object_assert pods "{{range.items}}{{$image_field}}:{{end}}" 'gcr.io/google_containers/serve_hostname:'
  # cleaning
  rm /tmp/tmp-editor.sh

  ## kubectl edit should work on Windows
  [ "$(EDITOR=cat kubectl edit pod/valid-pod 2>&1 | grep 'Edit cancelled')" ]
  [ "$(EDITOR=cat kubectl edit pod/valid-pod | grep 'name: valid-pod')" ]
  [ "$(EDITOR=cat kubectl edit --windows-line-endings pod/valid-pod | file - | grep CRLF)" ]
  [ ! "$(EDITOR=cat kubectl edit --windows-line-endings=false pod/valid-pod | file - | grep CRLF)" ]

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
  kubectl delete pods -l'name in (valid-pod-super-sayan)' --grace-period=0 "${kube_flags[@]}"
  # Post-condition: valid-pod POD doesn't exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''

  ### Create two PODs from 1 yaml file
  # Pre-condition: no POD exists
  create_and_use_new_namespace
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f docs/user-guide/multi-pod.yaml "${kube_flags[@]}"
  # Post-condition: valid-pod and redis-proxy PODs exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'redis-master:redis-proxy:'

  ### Delete two PODs from 1 yaml file
  # Pre-condition: redis-master and redis-proxy PODs exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'redis-master:redis-proxy:'
  # Command
  kubectl delete -f docs/user-guide/multi-pod.yaml "${kube_flags[@]}"
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
  [[ "$(kubectl get hpa.v1beta1.extensions frontend -o yaml "${kube_flags[@]}" | grep kubectl.kubernetes.io/last-applied-configuration)" ]]
  # Ensure we can interact with HPA objects in lists through both the extensions/v1beta1 and autoscaling/v1 APIs
  output_message=$(kubectl get hpa -o=jsonpath='{.items[0].apiVersion}' 2>&1 "${kube_flags[@]}")
  kube::test::if_has_string "${output_message}" 'autoscaling/v1'
  output_message=$(kubectl get hpa.extensions -o=jsonpath='{.items[0].apiVersion}' 2>&1 "${kube_flags[@]}")
  kube::test::if_has_string "${output_message}" 'extensions/v1beta1'
  output_message=$(kubectl get hpa.autoscaling -o=jsonpath='{.items[0].apiVersion}' 2>&1 "${kube_flags[@]}")
  kube::test::if_has_string "${output_message}" 'autoscaling/v1'
  # Clean up
  # Note that we should delete hpa first, otherwise it may fight with the rc reaper.
  kubectl delete hpa frontend "${kube_flags[@]}"
  kubectl delete rc  frontend "${kube_flags[@]}"

  ## kubectl create should not panic on empty string lists in a template
  ERROR_FILE="${KUBE_TEMP}/validation-error"
  kubectl create -f hack/testdata/invalid-rc-with-empty-args.yaml "${kube_flags[@]}" 2> "${ERROR_FILE}" || true
  # Post-condition: should get an error reporting the empty string
  if grep -q "unexpected nil value for field" "${ERROR_FILE}"; then
    kube::log::status "\"kubectl create with empty string list returns error as expected: $(cat ${ERROR_FILE})"
  else
    kube::log::status "\"kubectl create with empty string list returns unexpected error or non-error: $(cat ${ERROR_FILE})"
    exit 1
  fi
  rm "${ERROR_FILE}"

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

  ## kubectl run should create deployments or jobs
  # Pre-Condition: no Job exists
  kube::test::get_object_assert jobs "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl run pi --generator=job/v1beta1 "--image=$IMAGE_PERL" --restart=OnFailure -- perl -Mbignum=bpi -wle 'print bpi(20)' "${kube_flags[@]}"
  # Post-Condition: Job "pi" is created
  kube::test::get_object_assert jobs "{{range.items}}{{$id_field}}:{{end}}" 'pi:'
  # Clean up
  kubectl delete jobs pi "${kube_flags[@]}"
  # Command
  kubectl run pi --generator=job/v1 "--image=$IMAGE_PERL" --restart=OnFailure -- perl -Mbignum=bpi -wle 'print bpi(20)' "${kube_flags[@]}"
  # Post-Condition: Job "pi" is created
  kube::test::get_object_assert jobs "{{range.items}}{{$id_field}}:{{end}}" 'pi:'
  # Clean up
  kubectl delete jobs pi "${kube_flags[@]}"
  # Post-condition: no pods exist.
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Pre-Condition: no Deployment exists
  kube::test::get_object_assert deployment "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl run nginx "--image=$IMAGE_NGINX" --generator=deployment/v1beta1 "${kube_flags[@]}"
  # Post-Condition: Deployment "nginx" is created
  kube::test::get_object_assert deployment "{{range.items}}{{$id_field}}:{{end}}" 'nginx:'
  # Clean up
  kubectl delete deployment nginx "${kube_flags[@]}"

  ###############
  # Kubectl get #
  ###############

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

  #####################################
  # Third Party Resources             #
  #####################################
  create_and_use_new_namespace
  kubectl "${kube_flags[@]}" create -f - "${kube_flags[@]}" << __EOF__
{
  "kind": "ThirdPartyResource",
  "apiVersion": "extensions/v1beta1",
  "metadata": {
    "name": "foo.company.com"
  },
  "versions": [
    {
      "name": "v1"
    }
  ]
}
__EOF__

  # Post-Condition: assertion object exist
  kube::test::get_object_assert thirdpartyresources "{{range.items}}{{$id_field}}:{{end}}" 'foo.company.com:'

  kubectl "${kube_flags[@]}" create -f - "${kube_flags[@]}" << __EOF__
{
  "kind": "ThirdPartyResource",
  "apiVersion": "extensions/v1beta1",
  "metadata": {
    "name": "bar.company.com"
  },
  "versions": [
    {
      "name": "v1"
    }
  ]
}
__EOF__
  
  # Post-Condition: assertion object exist
  kube::test::get_object_assert thirdpartyresources "{{range.items}}{{$id_field}}:{{end}}" 'bar.company.com:foo.company.com:'

  kube::util::wait_for_url "http://127.0.0.1:${API_PORT}/apis/company.com/v1" "third party api"

  kube::util::wait_for_url "http://127.0.0.1:${API_PORT}/apis/company.com/v1/foos" "third party api Foo"

  kube::util::wait_for_url "http://127.0.0.1:${API_PORT}/apis/company.com/v1/bars" "third party api Bar"

  # Test that we can list this new third party resource (foos)
  kube::test::get_object_assert foos "{{range.items}}{{$id_field}}:{{end}}" ''

  # Test that we can list this new third party resource (bars)
  kube::test::get_object_assert bars "{{range.items}}{{$id_field}}:{{end}}" ''

  # Test that we can create a new resource of type Foo
  kubectl "${kube_flags[@]}" create -f - "${kube_flags[@]}" << __EOF__
{
  "kind": "Foo",
  "apiVersion": "company.com/v1",
  "metadata": {
    "name": "test"
  },
  "some-field": "field1",
  "other-field": "field2"
}
__EOF__

  # Test that we can list this new third party resource
  kube::test::get_object_assert foos "{{range.items}}{{$id_field}}:{{end}}" 'test:'

  # Delete the resource
  kubectl "${kube_flags[@]}" delete foos test

  # Make sure it's gone
  kube::test::get_object_assert foos "{{range.items}}{{$id_field}}:{{end}}" ''

  # Test that we can create a new resource of type Bar
  kubectl "${kube_flags[@]}" create -f - "${kube_flags[@]}" << __EOF__
{
  "kind": "Bar",
  "apiVersion": "company.com/v1",
  "metadata": {
    "name": "test"
  },
  "some-field": "field1",
  "other-field": "field2"
}
__EOF__

  # Test that we can list this new third party resource
  kube::test::get_object_assert bars "{{range.items}}{{$id_field}}:{{end}}" 'test:'

  # Delete the resource
  kubectl "${kube_flags[@]}" delete bars test

  # Make sure it's gone
  kube::test::get_object_assert bars "{{range.items}}{{$id_field}}:{{end}}" ''

  # teardown
  kubectl delete thirdpartyresources foo.company.com "${kube_flags[@]}"
  kubectl delete thirdpartyresources bar.company.com "${kube_flags[@]}"

  #################
  # Run cmd w img #
  #################

  # Test that a valid image reference value is provided as the value of --image in `kubectl run <name> --image`
  output_message=$(kubectl run test1 --image=validname)
  kube::test::if_has_string "${output_message}" 'deployment "test1" created'
  # test invalid image name
  output_message=$(! kubectl run test2 --image=InvalidImageName 2>&1)
  kube::test::if_has_string "${output_message}" 'error: Invalid image name "InvalidImageName": invalid reference format'


  #####################################
  # Recursive Resources via directory #
  #####################################

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
  # Post-condition: busybox0 & busybox1 PODs are edited, and since busybox2 is malformed, it should error
  kube::test::get_object_assert pods "{{range.items}}{{$image_field}}:{{end}}" 'prom/busybox:prom/busybox:'
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
  output_message=$(! kubectl delete -f hack/testdata/recursive/pod --recursive --grace-period=0 2>&1 "${kube_flags[@]}")
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
  kube::test::get_object_assert 'hpa busybox0' "{{$hpa_min_field}} {{$hpa_max_field}} {{$hpa_cpu_field}}" '1 2 <no value>'
  kube::test::get_object_assert 'hpa busybox1' "{{$hpa_min_field}} {{$hpa_max_field}} {{$hpa_cpu_field}}" '1 2 <no value>'
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
  output_message=$(! kubectl delete -f hack/testdata/recursive/rc --recursive --grace-period=0 2>&1 "${kube_flags[@]}")
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
  kube::test::get_object_assert deployment "{{range.items}}{{$deployment_image_field}}:{{end}}" "${IMAGE_NGINX}:${IMAGE_NGINX}:"
  ## Rollback the deployments to revision 1 recursively
  output_message=$(! kubectl rollout undo -f hack/testdata/recursive/deployment --recursive --to-revision=1 2>&1 "${kube_flags[@]}")
  # Post-condition: nginx0 & nginx1 should be a no-op, and since nginx2 is malformed, it should error
  kube::test::get_object_assert deployment "{{range.items}}{{$deployment_image_field}}:{{end}}" "${IMAGE_NGINX}:${IMAGE_NGINX}:"
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
  ! kubectl delete -f hack/testdata/recursive/deployment --recursive "${kube_flags[@]}" --grace-period=0
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
  kube::test::if_has_string "${output_message}" 'error when pausing "hack/testdata/recursive/rc/busybox.yaml'
  kube::test::if_has_string "${output_message}" 'error when pausing "hack/testdata/recursive/rc/rc/busybox.yaml'
  kube::test::if_has_string "${output_message}" "Object 'Kind' is missing"
  ## Attempt to resume the replication controllers recursively
  output_message=$(! kubectl rollout resume -f hack/testdata/recursive/rc --recursive 2>&1 "${kube_flags[@]}")
  # Post-condition: busybox0 & busybox1 should error as they are RC's, and since busybox2 is malformed, it should error
  kube::test::if_has_string "${output_message}" 'error when resuming "hack/testdata/recursive/rc/busybox.yaml'
  kube::test::if_has_string "${output_message}" 'error when resuming "hack/testdata/recursive/rc/rc/busybox.yaml'
  kube::test::if_has_string "${output_message}" "Object 'Kind' is missing"
  # Clean up
  ! kubectl delete -f hack/testdata/recursive/rc --recursive "${kube_flags[@]}" --grace-period=0
  sleep 1

  ##############
  # Namespaces #
  ##############

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

  ##############
  # Pods in Namespaces #
  ##############

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
  kubectl create "${kube_flags[@]}" --namespace=other -f docs/admin/limitrange/valid-pod.yaml
  # Post-condition: valid-pod POD is created
  kube::test::get_object_assert 'pods --namespace=other' "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'

  ### Delete POD valid-pod in specific namespace
  # Pre-condition: valid-pod POD exists
  kube::test::get_object_assert 'pods --namespace=other' "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'
  # Command
  kubectl delete "${kube_flags[@]}" pod --namespace=other valid-pod --grace-period=0
  # Post-condition: valid-pod POD doesn't exist
  kube::test::get_object_assert 'pods --namespace=other' "{{range.items}}{{$id_field}}:{{end}}" ''
  # Clean up
  kubectl delete namespace other

  ##############
  # Secrets #
  ##############

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
  # Pre-condition: no secret exists
  kube::test::get_object_assert 'secrets --namespace=test-secrets' "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  [[ "$(kubectl create secret generic test-secret --namespace=test-secrets --from-literal=key1=value1 --output=go-template --template=\"{{.metadata.name}}:\" | grep 'test-secret:')" ]]
  ## Clean-up
  kubectl delete secret test-secret --namespace=test-secrets
  # Clean up
  kubectl delete namespace test-secrets

  ######################
  # ConfigMap          #
  ######################

  kubectl create -f docs/user-guide/configmap/configmap.yaml
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

  ####################
  # Service Accounts #
  ####################

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

  #################
  # Pod templates #
  #################

  ### Create PODTEMPLATE
  # Pre-condition: no PODTEMPLATE
  kube::test::get_object_assert podtemplates "{{range.items}}{{.metadata.name}}:{{end}}" ''
  # Command
  kubectl create -f docs/user-guide/walkthrough/podtemplate.json "${kube_flags[@]}"
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


  ############
  # Services #
  ############
  # switch back to the default namespace
  kubectl config set-context "${CONTEXT}" --namespace=""
  kube::log::status "Testing kubectl(${version}:services)"

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

  ### Dump current redis-master service
  output_service=$(kubectl get service redis-master -o json --output-version=v1 "${kube_flags[@]}")

  ### Delete redis-master-service by id
  # Pre-condition: redis-master service exists
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:redis-master:'
  # Command
  kubectl delete service redis-master "${kube_flags[@]}"
  # Post-condition: Only the default kubernetes services exist
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:'

  ### Create redis-master-service from dumped JSON
  # Pre-condition: Only the default kubernetes services exist
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:'
  # Command
  echo "${output_service}" | kubectl create -f - "${kube_flags[@]}"
  # Post-condition: redis-master service is created
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:redis-master:'

  ### Create redis-master-${version}-test service
  # Pre-condition: redis-master-service service exists
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:redis-master:'
  # Command
  kubectl create -f - "${kube_flags[@]}" << __EOF__
{
  "kind": "Service",
  "apiVersion": "v1",
  "metadata": {
    "name": "service-${version}-test"
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
  # Post-condition: service-${version}-test service is created
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:redis-master:service-.*-test:'

  ### Identity
  kubectl get service "${kube_flags[@]}" service-${version}-test -o json | kubectl replace "${kube_flags[@]}" -f -

  ### Delete services by id
  # Pre-condition: service-${version}-test exists
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:redis-master:service-.*-test:'
  # Command
  kubectl delete service redis-master "${kube_flags[@]}"
  kubectl delete service "service-${version}-test" "${kube_flags[@]}"
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
  # Post-condition: Only the default kubernetes services exist
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:'


  ###########################
  # Replication controllers #
  ###########################

  kube::log::status "Testing kubectl(${version}:replicationcontrollers)"

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
  kube::test::describe_object_assert rc 'frontend' "Name:" "Image(s):" "Labels:" "Selector:" "Replicas:" "Pods Status:"
  # Describe command should print events information by default
  kube::test::describe_object_events_assert rc 'frontend'
  # Describe command should not print events information when show-events=false
  kube::test::describe_object_events_assert rc 'frontend' false
  # Describe command should print events information when show-events=true
  kube::test::describe_object_events_assert rc 'frontend' true
  # Describe command (resource only) should print detailed information
  kube::test::describe_resource_assert rc "Name:" "Name:" "Image(s):" "Labels:" "Selector:" "Replicas:" "Pods Status:"
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
  kubectl create -f docs/user-guide/job.yaml "${kube_flags[@]}"
  # Command
  kubectl scale --replicas=2 job/pi
  # Post-condition: 2 replicas for pi
  kube::test::get_object_assert 'job pi' "{{$job_parallelism_field}}" '2'
  # Clean-up
  kubectl delete job/pi "${kube_flags[@]}"

  ### Scale a deployment
  kubectl create -f docs/user-guide/deployment.yaml "${kube_flags[@]}"
  # Command
  kubectl scale --current-replicas=3 --replicas=1 deployment/nginx-deployment
  # Post-condition: 1 replica for nginx-deployment
  kube::test::get_object_assert 'deployment nginx-deployment' "{{$deployment_replicas}}" '1'
  # Clean-up
  kubectl delete deployment/nginx-deployment "${kube_flags[@]}"

  ### Expose a deployment as a service
  kubectl create -f docs/user-guide/deployment.yaml "${kube_flags[@]}"
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
  kubectl create -f docs/admin/limitrange/valid-pod.yaml "${kube_flags[@]}"
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
  output_message=$(kubectl expose -f docs/admin/high-availability/etcd.yaml --selector=test=etcd 2>&1 "${kube_flags[@]}")
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
  # autoscale 1~2 pods, CPU utilization 70%, rc specified by file, using old generator
  kubectl autoscale -f hack/testdata/frontend-controller.yaml "${kube_flags[@]}" --max=2 --cpu-percent=70 --generator=horizontalpodautoscaler/v1beta1
  kube::test::get_object_assert 'hpa frontend' "{{$hpa_min_field}} {{$hpa_max_field}} {{$hpa_cpu_field}}" '1 2 70'
  kubectl delete hpa frontend "${kube_flags[@]}"
  # autoscale 2~3 pods, no CPU utilization specified, rc specified by name
  kubectl autoscale rc frontend "${kube_flags[@]}" --min=2 --max=3
  kube::test::get_object_assert 'hpa frontend' "{{$hpa_min_field}} {{$hpa_max_field}} {{$hpa_cpu_field}}" '2 3 <no value>'
  kubectl delete hpa frontend "${kube_flags[@]}"
  # autoscale 2~3 pods, no CPU utilization specified, rc specified by name, using old generator
  kubectl autoscale rc frontend "${kube_flags[@]}" --min=2 --max=3 --generator=horizontalpodautoscaler/v1beta1
  kube::test::get_object_assert 'hpa frontend' "{{$hpa_min_field}} {{$hpa_max_field}} {{$hpa_cpu_field}}" '2 3 <no value>'
  kubectl delete hpa frontend "${kube_flags[@]}"
  # autoscale without specifying --max should fail
  ! kubectl autoscale rc frontend "${kube_flags[@]}"
  # Clean up
  kubectl delete rc frontend "${kube_flags[@]}"


  ######################
  # Deployments       #
  ######################

  ### Auto scale deployment
  # Pre-condition: no deployment exists
  kube::test::get_object_assert deployment "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f docs/user-guide/deployment.yaml "${kube_flags[@]}"
  kube::test::get_object_assert deployment "{{range.items}}{{$id_field}}:{{end}}" 'nginx-deployment:'
  # autoscale 2~3 pods, no CPU utilization specified
  kubectl-with-retry autoscale deployment nginx-deployment "${kube_flags[@]}" --min=2 --max=3
  kube::test::get_object_assert 'hpa nginx-deployment' "{{$hpa_min_field}} {{$hpa_max_field}} {{$hpa_cpu_field}}" '2 3 <no value>'
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
  kube::test::get_object_assert deployment "{{range.items}}{{$deployment_image_field}}:{{end}}" "${IMAGE_DEPLOYMENT_R1}:"
  # Rollback to revision 1 - should be no-op
  kubectl rollout undo deployment nginx --to-revision=1 "${kube_flags[@]}"
  kube::test::get_object_assert deployment "{{range.items}}{{$deployment_image_field}}:{{end}}" "${IMAGE_DEPLOYMENT_R1}:"
  # Update the deployment (revision 2)
  kubectl apply -f hack/testdata/deployment-revision2.yaml "${kube_flags[@]}"
  kube::test::get_object_assert deployment.extensions "{{range.items}}{{$deployment_image_field}}:{{end}}" "${IMAGE_DEPLOYMENT_R2}:"
  # Rollback to revision 1
  kubectl rollout undo deployment nginx --to-revision=1 "${kube_flags[@]}"
  sleep 1
  kube::test::get_object_assert deployment "{{range.items}}{{$deployment_image_field}}:{{end}}" "${IMAGE_DEPLOYMENT_R1}:"
  # Rollback to revision 1000000 - should be no-op
  kubectl rollout undo deployment nginx --to-revision=1000000 "${kube_flags[@]}"
  kube::test::get_object_assert deployment "{{range.items}}{{$deployment_image_field}}:{{end}}" "${IMAGE_DEPLOYMENT_R1}:"
  # Rollback to last revision
  kubectl rollout undo deployment nginx "${kube_flags[@]}"
  sleep 1
  kube::test::get_object_assert deployment "{{range.items}}{{$deployment_image_field}}:{{end}}" "${IMAGE_DEPLOYMENT_R2}:"
  # Pause the deployment
  kubectl-with-retry rollout pause deployment nginx "${kube_flags[@]}"
  # A paused deployment cannot be rolled back
  ! kubectl rollout undo deployment nginx "${kube_flags[@]}"
  # Resume the deployment
  kubectl-with-retry rollout resume deployment nginx "${kube_flags[@]}"
  # The resumed deployment can now be rolled back
  kubectl rollout undo deployment nginx "${kube_flags[@]}"
  # Clean up
  kubectl delete deployment nginx "${kube_flags[@]}"

  ### Set image of a deployment
  # Pre-condition: no deployment exists
  kube::test::get_object_assert deployment "{{range.items}}{{$id_field}}:{{end}}" ''
  # Create a deployment
  kubectl create -f hack/testdata/deployment-multicontainer.yaml "${kube_flags[@]}"
  kube::test::get_object_assert deployment "{{range.items}}{{$id_field}}:{{end}}" 'nginx-deployment:'
  kube::test::get_object_assert deployment "{{range.items}}{{$deployment_image_field}}:{{end}}" "${IMAGE_DEPLOYMENT_R1}:"
  kube::test::get_object_assert deployment "{{range.items}}{{$deployment_second_image_field}}:{{end}}" "${IMAGE_PERL}:"
  # Set the deployment's image
  kubectl set image deployment nginx-deployment nginx="${IMAGE_DEPLOYMENT_R2}" "${kube_flags[@]}"
  kube::test::get_object_assert deployment "{{range.items}}{{$deployment_image_field}}:{{end}}" "${IMAGE_DEPLOYMENT_R2}:"
  kube::test::get_object_assert deployment "{{range.items}}{{$deployment_second_image_field}}:{{end}}" "${IMAGE_PERL}:"
  # Set non-existing container should fail
  ! kubectl set image deployment nginx-deployment redis=redis "${kube_flags[@]}"
  # Set image of deployments without specifying name
  kubectl set image deployments --all nginx="${IMAGE_DEPLOYMENT_R1}" "${kube_flags[@]}"
  kube::test::get_object_assert deployment "{{range.items}}{{$deployment_image_field}}:{{end}}" "${IMAGE_DEPLOYMENT_R1}:"
  kube::test::get_object_assert deployment "{{range.items}}{{$deployment_second_image_field}}:{{end}}" "${IMAGE_PERL}:"
  # Set image of a deployment specified by file
  kubectl set image -f hack/testdata/deployment-multicontainer.yaml nginx="${IMAGE_DEPLOYMENT_R2}" "${kube_flags[@]}"
  kube::test::get_object_assert deployment "{{range.items}}{{$deployment_image_field}}:{{end}}" "${IMAGE_DEPLOYMENT_R2}:"
  kube::test::get_object_assert deployment "{{range.items}}{{$deployment_second_image_field}}:{{end}}" "${IMAGE_PERL}:"
  # Set image of a local file without talking to the server
  kubectl set image -f hack/testdata/deployment-multicontainer.yaml nginx="${IMAGE_DEPLOYMENT_R1}" "${kube_flags[@]}" --local -o yaml
  kube::test::get_object_assert deployment "{{range.items}}{{$deployment_image_field}}:{{end}}" "${IMAGE_DEPLOYMENT_R2}:"
  kube::test::get_object_assert deployment "{{range.items}}{{$deployment_second_image_field}}:{{end}}" "${IMAGE_PERL}:"
  # Set image of all containers of the deployment
  kubectl set image deployment nginx-deployment "*"="${IMAGE_DEPLOYMENT_R1}" "${kube_flags[@]}"
  kube::test::get_object_assert deployment "{{range.items}}{{$deployment_image_field}}:{{end}}" "${IMAGE_DEPLOYMENT_R1}:"
  kube::test::get_object_assert deployment "{{range.items}}{{$deployment_second_image_field}}:{{end}}" "${IMAGE_DEPLOYMENT_R1}:"
  # Clean up
  kubectl delete deployment nginx-deployment "${kube_flags[@]}"


  ######################
  # Replica Sets       #
  ######################

  kube::log::status "Testing kubectl(${version}:replicasets)"

  ### Create and stop a replica set, make sure it doesn't leak pods
  # Pre-condition: no replica set exists
  kube::test::get_object_assert rs "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f hack/testdata/frontend-replicaset.yaml "${kube_flags[@]}"
  kubectl delete rs frontend "${kube_flags[@]}"
  # Post-condition: no pods from frontend replica set
  kube::test::get_object_assert 'pods -l "tier=frontend"' "{{range.items}}{{$id_field}}:{{end}}" ''

  ### Create replica set frontend from YAML
  # Pre-condition: no replica set exists
  kube::test::get_object_assert rs "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f hack/testdata/frontend-replicaset.yaml "${kube_flags[@]}"
  # Post-condition: frontend replica set is created
  kube::test::get_object_assert rs "{{range.items}}{{$id_field}}:{{end}}" 'frontend:'
  # Describe command should print detailed information
  kube::test::describe_object_assert rs 'frontend' "Name:" "Image(s):" "Labels:" "Selector:" "Replicas:" "Pods Status:"
  # Describe command should print events information by default
  kube::test::describe_object_events_assert rs 'frontend'
  # Describe command should not print events information when show-events=false
  kube::test::describe_object_events_assert rs 'frontend' false
  # Describe command should print events information when show-events=true
  kube::test::describe_object_events_assert rs 'frontend' true
  # Describe command (resource only) should print detailed information
  kube::test::describe_resource_assert rs "Name:" "Name:" "Image(s):" "Labels:" "Selector:" "Replicas:" "Pods Status:"
  # Describe command should print events information by default
  kube::test::describe_resource_events_assert rs
  # Describe command should not print events information when show-events=false
  kube::test::describe_resource_events_assert rs false
  # Describe command should print events information when show-events=true
  kube::test::describe_resource_events_assert rs true

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
  kube::test::get_object_assert 'hpa frontend' "{{$hpa_min_field}} {{$hpa_max_field}} {{$hpa_cpu_field}}" '2 3 <no value>'
  kubectl delete hpa frontend "${kube_flags[@]}"
  # autoscale without specifying --max should fail
  ! kubectl autoscale rs frontend "${kube_flags[@]}"
  # Clean up
  kubectl delete rs frontend "${kube_flags[@]}"


  ######################
  # Lists              #
  ######################

  kube::log::status "Testing kubectl(${version}:lists)"

  ### Create a List with objects from multiple versions
  # Command
  kubectl create -f hack/testdata/list.yaml "${kube_flags[@]}"

  ### Delete the List with objects from multiple versions
  # Command
  kubectl delete service/list-service-test deployment/list-deployment-test


  ######################
  # Multiple Resources #
  ######################

  kube::log::status "Testing kubectl(${version}:multiple resources)"

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
    kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:'
    kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" ''
    # Command
    kubectl create -f "${file}" "${kube_flags[@]}"
    # Post-condition: mock service (and mock2) exists
    if [ "$has_svc" = true ]; then
      if [ "$two_svcs" = true ]; then
        kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:mock:mock2:'
      else
        kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:mock:'
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
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:'
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" ''

  # Command
  kubectl create -f https://raw.githubusercontent.com/kubernetes/kubernetes/master/hack/testdata/multi-resource-yaml.yaml "${kube_flags[@]}"

  # Post-condition: service(mock) and rc(mock) exist
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:mock:'
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" 'mock:'

  # Clean up
  kubectl delete -f https://raw.githubusercontent.com/kubernetes/kubernetes/master/hack/testdata/multi-resource-yaml.yaml "${kube_flags[@]}"

  # Post-condition: no service (other than default kubernetes services) or replication controller exists
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:'
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" ''


  ######################
  # Persistent Volumes #
  ######################

  ### Create and delete persistent volume examples
  # Pre-condition: no persistent volumes currently exist
  kube::test::get_object_assert pv "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f docs/user-guide/persistent-volumes/volumes/local-01.yaml "${kube_flags[@]}"
  kube::test::get_object_assert pv "{{range.items}}{{$id_field}}:{{end}}" 'pv0001:'
  kubectl delete pv pv0001 "${kube_flags[@]}"
  kubectl create -f docs/user-guide/persistent-volumes/volumes/local-02.yaml "${kube_flags[@]}"
  kube::test::get_object_assert pv "{{range.items}}{{$id_field}}:{{end}}" 'pv0002:'
  kubectl delete pv pv0002 "${kube_flags[@]}"
  kubectl create -f docs/user-guide/persistent-volumes/volumes/gce.yaml "${kube_flags[@]}"
  kube::test::get_object_assert pv "{{range.items}}{{$id_field}}:{{end}}" 'pv0003:'
  kubectl delete pv pv0003 "${kube_flags[@]}"
  # Post-condition: no PVs
  kube::test::get_object_assert pv "{{range.items}}{{$id_field}}:{{end}}" ''

  ############################
  # Persistent Volume Claims #
  ############################

  ### Create and delete persistent volume claim examples
  # Pre-condition: no persistent volume claims currently exist
  kube::test::get_object_assert pvc "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f docs/user-guide/persistent-volumes/claims/claim-01.yaml "${kube_flags[@]}"
  kube::test::get_object_assert pvc "{{range.items}}{{$id_field}}:{{end}}" 'myclaim-1:'
  kubectl delete pvc myclaim-1 "${kube_flags[@]}"

  kubectl create -f docs/user-guide/persistent-volumes/claims/claim-02.yaml "${kube_flags[@]}"
  kube::test::get_object_assert pvc "{{range.items}}{{$id_field}}:{{end}}" 'myclaim-2:'
  kubectl delete pvc myclaim-2 "${kube_flags[@]}"

  kubectl create -f docs/user-guide/persistent-volumes/claims/claim-03.json "${kube_flags[@]}"
  kube::test::get_object_assert pvc "{{range.items}}{{$id_field}}:{{end}}" 'myclaim-3:'
  kubectl delete pvc myclaim-3 "${kube_flags[@]}"
  # Post-condition: no PVCs
  kube::test::get_object_assert pvc "{{range.items}}{{$id_field}}:{{end}}" ''

  ############################
  # Storage Classes #
  ############################

  ### Create and delete storage class
  # Pre-condition: no storage classes currently exist
  kube::test::get_object_assert storageclass "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f - "${kube_flags[@]}" << __EOF__
{
  "kind": "StorageClass",
  "apiVersion": "extensions/v1beta1",
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
  kubectl delete storageclass storage-class-name "${kube_flags[@]}"
  # Post-condition: no storage classes
  kube::test::get_object_assert storageclass "{{range.items}}{{$id_field}}:{{end}}" ''

  #########
  # Nodes #
  #########

  kube::log::status "Testing kubectl(${version}:nodes)"

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
  kubectl create -f test/fixtures/pkg/kubectl/cmd/create/tokenreview.json --validate=false



  ########################
  # authorization.k8s.io #
  ########################

  # check remote authorization endpoint, kubectl doesn't actually display the returned object so this isn't super useful
  # but it proves that works
  kubectl create -f test/fixtures/pkg/kubectl/cmd/create/sar.json --validate=false

  SAR_RESULT_FILE="${KUBE_TEMP}/sar-result.json"
  curl -k -H "Content-Type:" http://localhost:8080/apis/authorization.k8s.io/v1beta1/subjectaccessreviews -XPOST -d @test/fixtures/pkg/kubectl/cmd/create/sar.json > "${SAR_RESULT_FILE}"
  if grep -q '"allowed": true' "${SAR_RESULT_FILE}"; then
    kube::log::status "\"authorization.k8s.io/subjectaccessreviews\" returns as expected: $(cat "${SAR_RESULT_FILE}")"
  else
    kube::log::status "\"authorization.k8s.io/subjectaccessreviews\" does not return as expected: $(cat "${SAR_RESULT_FILE}")"
    exit 1
  fi
  rm "${SAR_RESULT_FILE}"
 

  #####################
  # Retrieve multiple #
  #####################

  kube::log::status "Testing kubectl(${version}:multiget)"
  kube::test::get_object_assert 'nodes/127.0.0.1 service/kubernetes' "{{range.items}}{{$id_field}}:{{end}}" '127.0.0.1:kubernetes:'


  #####################
  # Resource aliasing #
  #####################

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

  ###########
  # Explain #
  ###########

  kube::log::status "Testing kubectl(${version}:explain)"
  kubectl explain pods
  # shortcuts work
  kubectl explain po
  kubectl explain po.status.message


  ###########
  # Swagger #
  ###########

  if [[ -n "${version}" ]]; then
    # Verify schema
    file="${KUBE_TEMP}/schema-${version}.json"
    curl -s "http://127.0.0.1:${API_PORT}/swaggerapi/api/${version}" > "${file}"
    [[ "$(grep "list of returned" "${file}")" ]]
    [[ "$(grep "List of pods" "${file}")" ]]
    [[ "$(grep "Watch for changes to the described resources" "${file}")" ]]
  fi

  #####################
  # Kubectl --sort-by #
  #####################

  ### sort-by should not panic if no pod exists
  # Pre-condition: no POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl get pods --sort-by="{metadata.name}"
  kubectl get pods --sort-by="{metadata.creationTimestamp}"

  ############################
  # Kubectl --all-namespaces #
  ############################

  # Pre-condition: the "default" namespace exists
  kube::test::get_object_assert namespaces "{{range.items}}{{if eq $id_field \\\"default\\\"}}{{$id_field}}:{{end}}{{end}}" 'default:'

  ### Create POD
  # Pre-condition: no POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create "${kube_flags[@]}" -f docs/admin/limitrange/valid-pod.yaml
  # Post-condition: valid-pod is created
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'

  ### Verify a specific namespace is ignored when all-namespaces is provided
  # Command
  kubectl get pods --all-namespaces --namespace=default

  ### Clean up
  # Pre-condition: valid-pod exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'
  # Command
  kubectl delete "${kube_flags[@]}" pod valid-pod --grace-period=0
  # Post-condition: valid-pod doesn't exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''

  kube::test::clear_all
}

runTests "v1"

kube::log::status "TEST PASSED"
