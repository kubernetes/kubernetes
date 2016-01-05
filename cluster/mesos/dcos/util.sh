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

# Example:
# export KUBERNETES_PROVIDER=mesos/dcos
# go run hack/e2e.go -v -up -check_cluster_size=false
# go run hack/e2e.go -v -test -check_version_skew=false
# go run hack/e2e.go -v -down

set -o errexit
set -o nounset
set -o pipefail
set -o errtrace

KUBE_ROOT=$(cd "$(dirname "${BASH_SOURCE}")/../../.." && pwd)
provider_root="${KUBE_ROOT}/cluster/${KUBERNETES_PROVIDER}"
dcos_config_file=~/.dcos/dcos.toml

source "${provider_root}/${KUBE_CONFIG_FILE-"config-default.sh"}"
source "${KUBE_ROOT}/cluster/common.sh"

# Read a toml config value
function toml-get {
  local f="$1"
  local section="$2"
  local key="$3"
  local def="${4:-}"

  sed -n "/^\[${section}\]/,/^\[.*\]/p" "${f}" |
    { grep "^[[:space:]*${key}[[:space:]]*=" || echo "${key}=${def}"; } |
    sed 's/.*=[[:space:]]*//;s/^"\(.*\)"[[:space:]]*$/\1/'
}

# Generate kubeconfig data for the created cluster.
function create-kubeconfig {
  local kubectl="${KUBE_ROOT}/cluster/kubectl.sh"

  export CONTEXT="${KUBERNETES_PROVIDER}"
  export KUBECONFIG=${KUBECONFIG:-$DEFAULT_KUBECONFIG}
  # KUBECONFIG determines the file we write to, but it may not exist yet
  if [[ ! -e "${KUBECONFIG}" ]]; then
    mkdir -p $(dirname "${KUBECONFIG}")
    touch "${KUBECONFIG}"
  fi

  dcos_ssl_verify="$(toml-get "${dcos_config_file}" core dcos_ssl_verify true)"
  local skip_tls_verify=false
  if [ "${dcos_ssl_verify}" == false ]; then
    skip_tls_verify=true
  fi

  "${kubectl}" config set-cluster "${CONTEXT}" --server="${KUBE_SERVER}" --insecure-skip-tls-verify="${skip_tls_verify}"
  "${kubectl}" config set-context "${CONTEXT}" --cluster="${CONTEXT}" --user="cluster-admin"
  "${kubectl}" config set-credentials cluster-admin --token=""
  "${kubectl}" config use-context "${CONTEXT}" --cluster="${CONTEXT}"

   echo "Wrote config for ${CONTEXT} to ${KUBECONFIG}" 1>&2
}

# Perform preparations required to run e2e tests
function prepare-e2e {
  echo "TODO: prepare-e2e" 1>&2
}

# Execute prior to running tests to build a release if required for env
function test-build-release {
  # Make a release
  export KUBERNETES_CONTRIB=mesos
  export KUBE_RELEASE_RUN_TESTS=N
  "${KUBE_ROOT}/build/release.sh"
}

# Must ensure that the following ENV vars are set
function detect-master {
  if [ ! -f "${dcos_config_file}" ]; then
    echo "ERROR: ${dcos_config_file} not found" 1>&2
    return 1
  fi
  dcos_url="$(toml-get "${dcos_config_file}" core dcos_url)"
  if [ -z "$dcos_url" ]; then
    echo "ERROR: dcos_url not set in ${dcos_config_file}" 1>&2
    return 1
  fi

  KUBE_MASTER_IP=""
  KUBE_SERVER="${dcos_url%/}/service/kubernetes/api"
  KUBE_MASTER_URL="${KUBE_SERVER}"

  echo "KUBE_MASTER_IP: $KUBE_MASTER_IP" 1>&2
  echo "KUBE_SERVER:    $KUBE_SERVER" 1>&2
}

# Get minion IP addresses and store in KUBE_NODE_IP_ADDRESSES[]
# These Mesos slaves MAY host Kublets,
# but might not have a Kublet running unless a kubernetes task has been scheduled on them.
function detect-nodes {
  KUBE_NODE_IP_ADDRESSES=(dummy)
  echo "KUBE_NODE_IP_ADDRESSES: [${KUBE_NODE_IP_ADDRESSES[*]}]" 1>&2
}

# Verify prereqs on host machine
function verify-prereqs {
  echo "Verifying required commands" 1>&2
  #hash dcos 2>/dev/null || { echo "Missing required command: dcos" 1>&2; exit 1; }
}

# Instantiate a kubernetes cluster.
function kube-up {
  echo "Expecting DCOS cluster already being up and Kubernetes deployed" 1>&2
  # await-health-check requires GNU timeout
  # apiserver hostname resolved by docker
  #cluster::mesos::docker::run_in_docker_test await-health-check "-t=${MESOS_DOCKER_API_TIMEOUT}" http://apiserver:8888/healthz

  detect-master
  detect-nodes
  create-kubeconfig

  # Wait for addons to deploy
  cluster::mesos::dcos::await_ready "kube-dns" "${MESOS_DCOS_ADDON_TIMEOUT}"
  cluster::mesos::dcos::await_ready "kube-ui" "${MESOS_DCOS_ADDON_TIMEOUT}"

  trap - EXIT
}

function validate-cluster {
  echo "Validating ${KUBERNETES_PROVIDER} cluster" 1>&2

  # Validate immediate cluster reachability and responsiveness
  echo "KubeDNS: $(cluster::mesos::dcos::addon_status 'kube-dns')"
  echo "KubeUI: $(cluster::mesos::dcos::addon_status 'kube-ui')"
}

# Delete a kubernetes cluster
function kube-down {
  echo "WARNING: leaving DCOS cluster and Kubernetes package running" 1>&2
}

function test-setup {
  echo "TODO: test-setup" 1>&2
}

# Execute after running tests to perform any required clean-up
function test-teardown {
  echo "test-teardown" 1>&2
  kube-down
}

## Below functions used by hack/e2e-suite/services.sh

# SSH to a node by name or IP ($1) and run a command ($2).
function ssh-to-node {
  echo "TODO: ssh-to-node" 1>&2
}

# Restart the kube-proxy on a node ($1)
function restart-kube-proxy {
  echo "TODO: restart-kube-proxy" 1>&2
}

# Restart the apiserver
function restart-apiserver {
  echo "TODO: restart-apiserver" 1>&2
}

# Waits for a kube-system pod (of the provided name) to have the phase/status "Running".
function cluster::mesos::dcos::await_ready {
  local pod_name="$1"
  local max_attempts="$2"
  local phase="Unknown"
  echo -n "${pod_name}: "
  local n=0
  until [ ${n} -ge ${max_attempts} ]; do
    phase=$(cluster::mesos::dcos::addon_status "${pod_name}")
    if [ "${phase}" == "Running" ]; then
      break
    fi
    echo -n "."
    n=$[$n+1]
    sleep 1
  done
  echo "${phase}"
  return $([ "${phase}" == "Running" ]; echo $?)
}

# Prints the status of the kube-system pod specified
function cluster::mesos::dcos::addon_status {
  local pod_name="$1"
  local kubectl="${KUBE_ROOT}/cluster/kubectl.sh"
  local phase=$("${kubectl}" get pods --namespace=kube-system -l k8s-app=${pod_name} -o template --template="{{(index .items 0).status.phase}}" 2>/dev/null)
  phase="${phase:-Unknown}"
  echo "${phase}"
}

