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
# export KUBERNETES_PROVIDER=mesos/docker
# go run hack/e2e.go -v -up -check_cluster_size=false
# go run hack/e2e.go -v -test -check_version_skew=false
# go run hack/e2e.go -v -down

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(cd "$(dirname "${BASH_SOURCE}")/../../.." && pwd)
provider_root="${KUBE_ROOT}/cluster/${KUBERNETES_PROVIDER}"
compose_file="${provider_root}/docker-compose.yml"

source "${provider_root}/${KUBE_CONFIG_FILE-"config-default.sh"}"
source "${KUBE_ROOT}/cluster/common.sh"
source "${provider_root}/util-ssl.sh"


# Run kubernetes scripts inside docker.
# This bypasses the need to set up network routing when running docker in a VM (e.g. boot2docker).
# Trap signals and kills the docker container for better signal handing
function cluster::mesos::docker::run_in_docker {
  entrypoint="$1"
  if [[ "${entrypoint}" = "./"* ]]; then
    # relative to project root
    entrypoint="/go/src/github.com/GoogleCloudPlatform/kubernetes/${entrypoint}"
  fi
  shift
  args="$@"

  container_id=$(
    docker run \
      -d \
      -e "KUBERNETES_PROVIDER=${KUBERNETES_PROVIDER}" \
      -v "${KUBE_ROOT}:/go/src/github.com/GoogleCloudPlatform/kubernetes" \
      -v "$(dirname ${KUBECONFIG}):/root/.kube" \
      -v "/var/run/docker.sock:/var/run/docker.sock" \
      --link docker_mesosmaster1_1:mesosmaster1 \
      --link docker_mesosslave1_1:mesosslave1 \
      --link docker_mesosslave2_1:mesosslave2 \
      --link docker_apiserver_1:apiserver \
      --entrypoint="${entrypoint}" \
      mesosphere/kubernetes-mesos-test \
      ${args}
  )

  docker logs -f "${container_id}" &

  # trap and kill for better signal handing
  trap 'echo "Killing container ${container_id}" 1>&2 && docker kill ${container_id}' INT TERM
  exit_status=$(docker wait "${container_id}")
  trap - INT TERM

  if [ "$exit_status" != 0 ]; then
    echo "Exited ${exit_status}" 1>&2
  fi

  docker rm -f "${container_id}" > /dev/null

  return "${exit_status}"
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

  local token="$(cut -d, -f1 ${provider_root}/certs/apiserver/token-users)"
  "${kubectl}" config set-cluster "${CONTEXT}" --server="${KUBE_SERVER}" --certificate-authority="${provider_root}/certs/root-ca.crt"
  "${kubectl}" config set-context "${CONTEXT}" --cluster="${CONTEXT}" --user="cluster-admin"
  "${kubectl}" config set-credentials cluster-admin --token="${token}"
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
  #  echo "KUBE_MASTER: $KUBE_MASTER" 1>&2

  docker_id=$(docker ps --filter="name=docker_apiserver" --quiet)
  if [[ "${docker_id}" == *'\n'* ]]; then
    echo "ERROR: Multiple API Servers running in docker" 1>&2
    return 1
  fi

  master_ip=$(docker inspect --format="{{.NetworkSettings.IPAddress}}" "${docker_id}")
  master_port=6443

  KUBE_MASTER_IP="${master_ip}:${master_port}"
  KUBE_SERVER="https://${KUBE_MASTER_IP}"

  echo "KUBE_MASTER_IP: $KUBE_MASTER_IP" 1>&2
}

# Get minion IP addresses and store in KUBE_MINION_IP_ADDRESSES[]
# These Mesos slaves MAY host Kublets,
# but might not have a Kublet running unless a kubernetes task has been scheduled on them.
function detect-minions {
  docker_ids=$(docker ps --filter="name=docker_mesosslave" --quiet)
  while read -r docker_id; do
    minion_ip=$(docker inspect --format="{{.NetworkSettings.IPAddress}}" "${docker_id}")
    KUBE_MINION_IP_ADDRESSES+=("${minion_ip}")
  done <<< "$docker_ids"
  echo "KUBE_MINION_IP_ADDRESSES: [${KUBE_MINION_IP_ADDRESSES[*]}]" 1>&2
}

# Verify prereqs on host machine
function verify-prereqs {
  echo "TODO: verify-prereqs" 1>&2
  # Verify that docker, docker-compose, and jq exist
  # Verify that all the right docker images exist:
  # mesosphere/kubernetes-mesos-test, etc.
}

# Instantiate a kubernetes cluster.
function kube-up {
  # TODO: version images (k8s version, git sha, and dirty state) to avoid re-building them every time.
  if [ "${MESOS_DOCKER_SKIP_BUILD:-false}" != "true" ]; then
    echo "Building Docker images" 1>&2
    "${provider_root}/km/build.sh"
    "${provider_root}/test/build.sh"
  fi

  local certdir="${provider_root}/certs"

  # create mount volume dirs
  local apiserver_certs_dir="${certdir}/apiserver"
  local controller_certs_dir="${certdir}/controller"
  # clean certs directory from previous runs
  if [ -d "${apiserver_certs_dir}" ]; then
    rm -rf "${apiserver_certs_dir}"/*
  fi
  mkdir -p "${apiserver_certs_dir}" "${controller_certs_dir}"

  echo "Creating root certificate authority" 1>&2
  cluster::mesos::docker::create_root_certificate_authority "${certdir}"
  cp "${certdir}/root-ca.crt" "${controller_certs_dir}/"

  echo "Creating service-account rsa key" 1>&2
  cluster::mesos::docker::create_rsa_key "${certdir}/service-accounts.key"
  cp "${certdir}/service-accounts.key" "${apiserver_certs_dir}/"
  cp "${certdir}/service-accounts.key" "${controller_certs_dir}/"

  echo "Creating cluster-admin token user" 1>&2
  cluster::mesos::docker::create_token_user "cluster-admin" > "${apiserver_certs_dir}/token-users"

  echo "Creating admin basic auth user" 1>&2
  cluster::mesos::docker::create_basic_user "admin" "admin" > "${apiserver_certs_dir}/basic-users"

  # log dump on failure
  trap 'cluster::mesos::docker::dump_logs' ERR

  echo "Starting ${KUBERNETES_PROVIDER} cluster" 1>&2
  docker-compose -f "${compose_file}" up -d

  detect-master
  detect-minions

  # The apiserver is waiting for its certificate, which depends on the IP docker chose.
  echo "Creating apiserver certificate" 1>&2
  local apiserer_ip="$(cut -f1 -d: <<<${KUBE_MASTER_IP})"
  local apiservice_ip="10.10.10.1"
  cluster::mesos::docker::create_apiserver_cert \
    "${certdir}" "${apiserver_certs_dir}" "${apiserer_ip}" "${apiservice_ip}"

  # KUBECONFIG needs to exist before run_in_docker mounts it, otherwise it will be owned by root
  create-kubeconfig

  # await-health-check could be run locally, but it would require GNU timeout installed on mac...
  # "${provider_root}/common/bin/await-health-check" -t=120 ${KUBE_SERVER}/healthz
  cluster::mesos::docker::run_in_docker await-health-check -t=120 http://apiserver:8888/healthz

  echo "Deploying Addons" 1>&2
  KUBE_SERVER=${KUBE_SERVER} "${provider_root}/deploy-addons.sh"

  # Wait for addons to deploy
  cluster::mesos::docker::await_ready "kube-dns"
  cluster::mesos::docker::await_ready "kube-ui"
}

function validate-cluster {
  echo "Validating ${KUBERNETES_PROVIDER} cluster" 1>&2

  # Do not validate cluster size. There will be zero k8s minions until a pod is created.
  # TODO(karlkfi): use componentstatuses or equivalent when it supports non-localhost core components

  # Validate immediate cluster reachability and responsiveness
  echo "KubeDNS: $(cluster::mesos::docker::addon_status 'kube-dns')"
  echo "KubeUI: $(cluster::mesos::docker::addon_status 'kube-ui')"
}

# Delete a kubernetes cluster
function kube-down {
  echo "Stopping ${KUBERNETES_PROVIDER} cluster" 1>&2
  # Since restoring a stopped cluster is not yet supported, use the nuclear option
  docker-compose -f "${compose_file}" kill
  docker-compose -f "${compose_file}" rm -f
}

function test-setup {
  echo "Building required docker images" 1>&2
  "${KUBE_ROOT}/cluster/mesos/docker/km/build.sh"
  "${KUBE_ROOT}/cluster/mesos/docker/test/build.sh"
  "${KUBE_ROOT}/cluster/mesos/docker/mesos-slave/build.sh"
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
function cluster::mesos::docker::await_ready {
  local pod_name=$1
  local max_attempts=60
  local phase="Unknown"
  echo -n "${pod_name}: "
  local n=0
  until [ ${n} -ge ${max_attempts} ]; do
    phase=$(cluster::mesos::docker::addon_status "${pod_name}")
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
function cluster::mesos::docker::addon_status {
  local pod_name=$1
  local kubectl="${KUBE_ROOT}/cluster/kubectl.sh"
  local phase=$("${kubectl}" get pods --namespace=kube-system -l k8s-app=${pod_name} -o template --template="{{(index .items 0).status.phase}}" 2>/dev/null)
  phase="${phase:-Unknown}"
  echo "${phase}"
}

function cluster::mesos::docker::dump_logs {
  local log_dir="${provider_root}/logs"
  echo "Dumping logs to '${log_dir}'" 1>&2
  mkdir -p "${log_dir}"
  while read name; do
    docker logs "${name}" &> "${log_dir}/${name}.log"
  done < <(docker-compose -f "${compose_file}" ps -q | xargs docker inspect --format '{{.Name}}')
}
