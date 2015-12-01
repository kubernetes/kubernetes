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
set -o errtrace

KUBE_ROOT=$(cd "$(dirname "${BASH_SOURCE}")/../../.." && pwd)
provider_root="${KUBE_ROOT}/cluster/${KUBERNETES_PROVIDER}"

source "${provider_root}/${KUBE_CONFIG_FILE-"config-default.sh"}"
source "${KUBE_ROOT}/cluster/common.sh"
source "${provider_root}/common/bin/util-ssl.sh"

# Execute a docker-compose command with the default environment and compose file.
function cluster::mesos::docker::docker_compose {
  local params="$@"

  # All vars required to be set
  declare -a env_vars=(
    "KUBE_KEYGEN_TIMEOUT"
    "MESOS_DOCKER_ETCD_TIMEOUT"
    "MESOS_DOCKER_MESOS_TIMEOUT"
    "MESOS_DOCKER_API_TIMEOUT"
    "MESOS_DOCKER_ADDON_TIMEOUT"
    "MESOS_DOCKER_WORK_DIR"
    "DOCKER_DAEMON_ARGS"
  )

  (
    for var_name in "${env_vars[@]}"; do
      export ${var_name}="${!var_name}"
    done

    docker-compose -f "${provider_root}/docker-compose.yml" ${params}
  )
}

# Pull the images from a docker compose file, if they're not already cached.
# This avoid slow remote calls from `docker-compose pull` which delegates
# to `docker pull` which always hits the remote docker repo, even if the image
# is already cached.
function cluster::mesos::docker::docker_compose_lazy_pull {
  for img in $(grep '^\s*image:\s' "${provider_root}/docker-compose.yml" | sed 's/[ \t]*image:[ \t]*//'); do
    read repo tag <<<$(echo "${img} "| sed 's/:/ /')
    if [ -z "${tag}" ]; then
      tag="latest"
    fi
    if ! docker images "${repo}" | awk '{print $2;}' | grep -q "${tag}"; then
      docker pull "${img}"
    fi
  done
}

# Run kubernetes scripts inside docker.
# This bypasses the need to set up network routing when running docker in a VM (e.g. boot2docker).
# Trap signals and kills the docker container for better signal handing
function cluster::mesos::docker::run_in_docker_test {
  local entrypoint="$1"
  if [[ "${entrypoint}" = "./"* ]]; then
    # relative to project root
    entrypoint="/go/src/github.com/GoogleCloudPlatform/kubernetes/${entrypoint}"
  fi
  shift
  local args="$@"

  # only mount KUBECONFIG if it exists, otherwise the directory will be created/owned by root
  kube_config_mount=""
  if [ -n "${KUBECONFIG:-}" ] && [ -e "${KUBECONFIG}" ]; then
    kube_config_mount="-v \"$(dirname ${KUBECONFIG}):/root/.kube\""
  fi

  container_id=$(
    docker run \
      -d \
      -e "KUBERNETES_PROVIDER=${KUBERNETES_PROVIDER}" \
      -v "${KUBE_ROOT}:/go/src/github.com/GoogleCloudPlatform/kubernetes" \
      ${kube_config_mount} \
      -v "/var/run/docker.sock:/var/run/docker.sock" \
      --link docker_mesosmaster1_1:mesosmaster1 \
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

# Run kube-cagen.sh inside docker.
# Creating and signing in the same environment avoids a subject comparison string_mask issue.
function cluster::mesos::docker::run_in_docker_cagen {
  local out_dir="$1"

  container_id=$(
    docker run \
      -d \
      -v "${out_dir}:/var/run/kubernetes/auth" \
      --entrypoint="kube-cagen.sh" \
      mesosphere/kubernetes-mesos-keygen \
      "/var/run/kubernetes/auth"
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
  local -r auth_dir="${MESOS_DOCKER_WORK_DIR}/auth"
  local kubectl="${KUBE_ROOT}/cluster/kubectl.sh"

  export CONTEXT="${KUBERNETES_PROVIDER}"
  export KUBECONFIG=${KUBECONFIG:-$DEFAULT_KUBECONFIG}
  # KUBECONFIG determines the file we write to, but it may not exist yet
  if [[ ! -e "${KUBECONFIG}" ]]; then
    mkdir -p $(dirname "${KUBECONFIG}")
    touch "${KUBECONFIG}"
  fi

  local token="$(cut -d, -f1 ${auth_dir}/token-users)"
  "${kubectl}" config set-cluster "${CONTEXT}" --server="${KUBE_SERVER}" --certificate-authority="${auth_dir}/root-ca.crt"
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

  local docker_id=$(docker ps --filter="name=docker_apiserver" --quiet)
  if [[ "${docker_id}" == *'\n'* ]]; then
    echo "ERROR: Multiple API Servers running" 1>&2
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
  local docker_ids=$(docker ps --filter="name=docker_mesosslave" --quiet)
  if [ -z "${docker_ids}" ]; then
    echo "ERROR: Mesos slave(s) not running" 1>&2
    return 1
  fi
  while read -r docker_id; do
    local minion_ip=$(docker inspect --format="{{.NetworkSettings.IPAddress}}" "${docker_id}")
    KUBE_MINION_IP_ADDRESSES+=("${minion_ip}")
  done <<< "$docker_ids"
  echo "KUBE_MINION_IP_ADDRESSES: [${KUBE_MINION_IP_ADDRESSES[*]}]" 1>&2
}

# Verify prereqs on host machine
function verify-prereqs {
  echo "Verifying required commands" 1>&2
  hash docker 2>/dev/null || { echo "Missing required command: docker" 1>&2; exit 1; }
  hash docker 2>/dev/null || { echo "Missing required command: docker-compose" 1>&2; exit 1; }
}

# Initialize
function cluster::mesos::docker::init_auth {
  local -r auth_dir="${MESOS_DOCKER_WORK_DIR}/auth"

  #TODO(karlkfi): reuse existing credentials/certs/keys
  # Nuke old auth
  echo "Creating Auth Dir: ${auth_dir}" 1>&2
  mkdir -p "${auth_dir}"
  rm -rf "${auth_dir}"/*

  echo "Creating Certificate Authority" 1>&2
  cluster::mesos::docker::run_in_docker_cagen "${auth_dir}"

  echo "Creating Service-Account RSA Key" 1>&2
  cluster::mesos::docker::create_rsa_key "${auth_dir}/service-accounts.key"

  echo "Creating User Accounts" 1>&2
  cluster::mesos::docker::create_token_user "cluster-admin" > "${auth_dir}/token-users"
  echo "Token Users: ${auth_dir}/token-users" 1>&2
  cluster::mesos::docker::create_basic_user "admin" "admin" > "${auth_dir}/basic-users"
  echo "Basic-Auth Users: ${auth_dir}/basic-users" 1>&2
}

# Instantiate a kubernetes cluster.
function kube-up {
  # Nuke old mesos-slave workspaces
  local work_dir="${MESOS_DOCKER_WORK_DIR}/mesosslave"
  echo "Creating Mesos Work Dir: ${work_dir}" 1>&2
  mkdir -p "${work_dir}"
  rm -rf "${work_dir}"/*

  # Nuke old logs
  local -r log_dir="${MESOS_DOCKER_WORK_DIR}/log"
  mkdir -p "${log_dir}"
  rm -rf "${log_dir}"/*

  # Pull before `docker-compose up` to avoid timeouts caused by slow pulls during deployment.
  echo "Pulling Docker images" 1>&2
  cluster::mesos::docker::docker_compose_lazy_pull

  if [ "${MESOS_DOCKER_SKIP_BUILD}" != "true" ]; then
    echo "Building Docker images" 1>&2
    # TODO: version images (k8s version, git sha, and dirty state) to avoid re-building them every time.
    "${provider_root}/km/build.sh"
    "${provider_root}/test/build.sh"
    "${provider_root}/keygen/build.sh"
  fi

  cluster::mesos::docker::init_auth

  # Dump logs on premature exit (errexit triggers exit).
  # Trap EXIT instead of ERR, because ERR can trigger multiple times with errtrace enabled.
  trap "cluster::mesos::docker::dump_logs '${log_dir}'" EXIT

  echo "Starting ${KUBERNETES_PROVIDER} cluster" 1>&2
  cluster::mesos::docker::docker_compose up -d
  echo "Scaling ${KUBERNETES_PROVIDER} cluster to ${NUM_MINIONS} slaves"
  cluster::mesos::docker::docker_compose scale mesosslave=${NUM_MINIONS}

  # await-health-check requires GNU timeout
  # apiserver hostname resolved by docker
  cluster::mesos::docker::run_in_docker_test await-health-check "-t=${MESOS_DOCKER_API_TIMEOUT}" http://apiserver:8888/healthz

  detect-master
  detect-minions
  create-kubeconfig

  echo "Deploying Addons" 1>&2
  KUBE_SERVER=${KUBE_SERVER} "${provider_root}/deploy-addons.sh"

  # Wait for addons to deploy
  cluster::mesos::docker::await_ready "kube-dns" "${MESOS_DOCKER_ADDON_TIMEOUT}"
  cluster::mesos::docker::await_ready "kube-ui" "${MESOS_DOCKER_ADDON_TIMEOUT}"

  trap - EXIT
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
  if [ "${MESOS_DOCKER_DUMP_LOGS}" == "true" ]; then
    cluster::mesos::docker::dump_logs "${MESOS_DOCKER_WORK_DIR}/log"
  fi
  echo "Stopping ${KUBERNETES_PROVIDER} cluster" 1>&2
  # Since restoring a stopped cluster is not yet supported, use the nuclear option
  cluster::mesos::docker::docker_compose kill
  cluster::mesos::docker::docker_compose rm -f
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
function cluster::mesos::docker::await_ready {
  local pod_name="$1"
  local max_attempts="$2"
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
  local pod_name="$1"
  local kubectl="${KUBE_ROOT}/cluster/kubectl.sh"
  local phase=$("${kubectl}" get pods --namespace=kube-system -l k8s-app=${pod_name} -o template --template="{{(index .items 0).status.phase}}" 2>/dev/null)
  phase="${phase:-Unknown}"
  echo "${phase}"
}

function cluster::mesos::docker::dump_logs {
  local out_dir="$1"
  echo "Dumping logs to '${out_dir}'" 1>&2
  mkdir -p "${out_dir}"
  while read name; do
    docker logs "${name}" &> "${out_dir}/${name}.log"
  done < <(cluster::mesos::docker::docker_compose ps -q | xargs docker inspect --format '{{.Name}}')
}
