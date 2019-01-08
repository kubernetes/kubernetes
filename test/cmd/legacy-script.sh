#!/usr/bin/env bash

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
source "${KUBE_ROOT}/test/cmd/apply.sh"
source "${KUBE_ROOT}/test/cmd/apps.sh"
source "${KUBE_ROOT}/test/cmd/authorization.sh"
source "${KUBE_ROOT}/test/cmd/batch.sh"
source "${KUBE_ROOT}/test/cmd/certificate.sh"
source "${KUBE_ROOT}/test/cmd/core.sh"
source "${KUBE_ROOT}/test/cmd/crd.sh"
source "${KUBE_ROOT}/test/cmd/create.sh"
source "${KUBE_ROOT}/test/cmd/diff.sh"
source "${KUBE_ROOT}/test/cmd/discovery.sh"
source "${KUBE_ROOT}/test/cmd/generic-resources.sh"
source "${KUBE_ROOT}/test/cmd/get.sh"
source "${KUBE_ROOT}/test/cmd/initializers.sh"
source "${KUBE_ROOT}/test/cmd/kubeconfig.sh"
source "${KUBE_ROOT}/test/cmd/node-management.sh"
source "${KUBE_ROOT}/test/cmd/old-print.sh"
source "${KUBE_ROOT}/test/cmd/plugins.sh"
source "${KUBE_ROOT}/test/cmd/proxy.sh"
source "${KUBE_ROOT}/test/cmd/rbac.sh"
source "${KUBE_ROOT}/test/cmd/request-timeout.sh"
source "${KUBE_ROOT}/test/cmd/run.sh"
source "${KUBE_ROOT}/test/cmd/save-config.sh"
source "${KUBE_ROOT}/test/cmd/storage.sh"
source "${KUBE_ROOT}/test/cmd/template-output.sh"
source "${KUBE_ROOT}/test/cmd/version.sh"


ETCD_HOST=${ETCD_HOST:-127.0.0.1}
ETCD_PORT=${ETCD_PORT:-2379}
API_PORT=${API_PORT:-8080}
SECURE_API_PORT=${SECURE_API_PORT:-6443}
API_HOST=${API_HOST:-127.0.0.1}
KUBELET_HEALTHZ_PORT=${KUBELET_HEALTHZ_PORT:-10248}
CTLRMGR_PORT=${CTLRMGR_PORT:-10252}
PROXY_HOST=127.0.0.1 # kubectl only serves on localhost.

IMAGE_NGINX="k8s.gcr.io/nginx:1.7.9"
IMAGE_DEPLOYMENT_R1="k8s.gcr.io/nginx:test-cmd"  # deployment-revision1.yaml
IMAGE_DEPLOYMENT_R2="$IMAGE_NGINX"  # deployment-revision2.yaml
IMAGE_PERL="k8s.gcr.io/perl"
IMAGE_PAUSE_V2="k8s.gcr.io/pause:2.0"
IMAGE_DAEMONSET_R2="k8s.gcr.io/pause:latest"
IMAGE_DAEMONSET_R2_2="k8s.gcr.io/nginx:test-cmd"  # rollingupdate-daemonset-rv2.yaml
IMAGE_STATEFULSET_R1="k8s.gcr.io/nginx-slim:0.7"
IMAGE_STATEFULSET_R2="k8s.gcr.io/nginx-slim:0.8"

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
customresourcedefinitions="customresourcedefinitions"
daemonsets="daemonsets"
controllerrevisions="controllerrevisions"
job="jobs"


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
      foundError="${foundError}""${name}"", "
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
    kubeout=`kubectl get po -l $1 --output=go-template --template='{{range.items}}{{.metadata.name}}{{end}}' --sort-by metadata.name "${kube_flags[@]}"`
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
  kube::util::ensure-gnu-sed

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

# Runs all kubectl tests.
# Requires an env var SUPPORTED_RESOURCES which is a comma separated list of
# resources for which tests should be run.
runTests() {
  foundError=""

  if [ -z "${SUPPORTED_RESOURCES:-}" ]; then
    echo "Need to set SUPPORTED_RESOURCES env var. It is a list of resources that are supported and hence should be tested. Set it to (*) to test all resources"
    exit 1
  fi
  kube::log::status "Checking kubectl version"
  kubectl version

  # Generate a random namespace name, based on the current time (to make
  # debugging slightly easier) and a random number. Don't use `date +%N`
  # because that doesn't work on OSX.
  create_and_use_new_namespace() {
    local ns_name
    ns_name="namespace-$(date +%s)-${RANDOM}"
    kube::log::status "Creating namespace ${ns_name}"
    kubectl create namespace "${ns_name}"
    kubectl config set-context "${CONTEXT}" --namespace="${ns_name}"
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
  template_labels=".spec.template.metadata.labels.name"
  statefulset_replicas_field=".spec.replicas"
  statefulset_observed_generation=".status.observedGeneration"
  job_parallelism_field=".spec.parallelism"
  deployment_replicas=".spec.replicas"
  secret_data=".data"
  secret_type=".type"
  change_cause_annotation='.*kubernetes.io/change-cause.*'
  pdb_min_available=".spec.minAvailable"
  pdb_max_unavailable=".spec.maxUnavailable"
  generation_field=".metadata.generation"
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

  # find all resources
  kubectl "${kube_flags[@]}" api-resources
  # find all namespaced resources that support list by name and get them
  kubectl "${kube_flags[@]}" api-resources --verbs=list --namespaced -o name | xargs -n 1 kubectl "${kube_flags[@]}" get -o name

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
    record_command run_kubectl_apply_tests
    record_command run_kubectl_run_tests
    record_command run_kubectl_create_filter_tests
  fi

  if kube::test::if_supports_resource "${deployments}" ; then
    record_command run_kubectl_apply_deployments_tests
  fi

  ################
  # Kubectl diff #
  ################
  record_command run_kubectl_diff_tests
  record_command run_kubectl_diff_same_names

  ###############
  # Kubectl get #
  ###############

  if kube::test::if_supports_resource "${pods}" ; then
    record_command run_kubectl_get_tests
    record_command run_kubectl_old_print_tests
  fi


  ######################
  # Create             #
  ######################
  if kube::test::if_supports_resource "${secrets}" ; then
    record_command run_create_secret_tests
  fi

  ##################
  # Global timeout #
  ##################

  if kube::test::if_supports_resource "${pods}" ; then
    record_command run_kubectl_request_timeout_tests
  fi

  #####################################
  # CustomResourceDefinitions         #
  #####################################

  # customresourcedefinitions cleanup after themselves.
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

  ####################
  # Job              #
  ####################

  if kube::test::if_supports_resource "${job}" ; then
    record_command run_job_tests
    record_command run_create_job_tests
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
    kube::test::if_has_string "${output_message}" "subresource can not be used with NonResourceURL"

    output_message=$(kubectl auth can-i list jobs.batch/bar -n foo --quiet 2>&1 "${kube_flags[@]}")
    kube::test::if_empty_string "${output_message}"

    output_message=$(kubectl auth can-i get pods --subresource=log 2>&1 "${kube_flags[@]}"; echo $?)
    kube::test::if_has_string "${output_message}" '0'

    output_message=$(kubectl auth can-i get pods --subresource=log --quiet 2>&1 "${kube_flags[@]}"; echo $?)
    kube::test::if_has_string "${output_message}" '0'
  fi

  # kubectl auth reconcile
  if kube::test::if_supports_resource "${clusterroles}" ; then
    kubectl auth reconcile "${kube_flags[@]}" -f test/fixtures/pkg/kubectl/cmd/auth/rbac-resource-plus.yaml
    kube::test::get_object_assert 'rolebindings -n some-other-random -l test-cmd=auth' "{{range.items}}{{$id_field}}:{{end}}" 'testing-RB:'
    kube::test::get_object_assert 'roles -n some-other-random -l test-cmd=auth' "{{range.items}}{{$id_field}}:{{end}}" 'testing-R:'
    kube::test::get_object_assert 'clusterrolebindings -l test-cmd=auth' "{{range.items}}{{$id_field}}:{{end}}" 'testing-CRB:'
    kube::test::get_object_assert 'clusterroles -l test-cmd=auth' "{{range.items}}{{$id_field}}:{{end}}" 'testing-CR:'

    failure_message=$(! kubectl auth reconcile "${kube_flags[@]}" -f test/fixtures/pkg/kubectl/cmd/auth/rbac-v1beta1.yaml 2>&1 )
    kube::test::if_has_string "${failure_message}" 'only rbac.authorization.k8s.io/v1 is supported'

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
    if kube::test::if_supports_resource "${nodes}" ; then
      record_command run_kubectl_all_namespace_tests
    fi
  fi

  ######################
  # kubectl --template #
  ######################

  if kube::test::if_supports_resource "${pods}" ; then
    record_command run_template_output_tests
  fi

  ################
  # Certificates #
  ################

  if kube::test::if_supports_resource "${csr}" ; then
    record_command run_certificates_tests
  fi

  ######################
  # Cluster Management #
  ######################
  if kube::test::if_supports_resource "${nodes}" ; then
    record_command run_cluster_management_tests
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

  if [[ -n "${foundError}" ]]; then
    echo "FAILED TESTS: ""${foundError}"
    exit 1
  fi
}
