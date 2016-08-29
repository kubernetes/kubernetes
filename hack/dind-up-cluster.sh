#!/bin/bash

# Copyright 2016 The Kubernetes Authors
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
set -o errtrace

DIND_ROOT="$(dirname "${BASH_SOURCE}")"
KUBE_ROOT=$(cd "${DIND_ROOT}/.." && pwd)

# Execute a docker-compose command with the default environment and compose file.
function dind::docker_compose {
  local params="$@"

  # All vars required to be set
  declare -a env_vars=(
    "DOCKER_IN_DOCKER_WORK_DIR"
    "APISERVER_SERVICE_IP"
    "SERVICE_CIDR"
    "DNS_SERVER_IP"
    "DNS_DOMAIN"
  )

  (
    for var_name in "${env_vars[@]}"; do
      export ${var_name}="${!var_name}"
    done

    export DOCKER_IN_DOCKER_STORAGE_DIR=${DOCKER_IN_DOCKER_STORAGE_DIR:-${DOCKER_IN_DOCKER_WORK_DIR}/storage}

    docker-compose -p dind -f "${KUBE_ROOT}/hack/dind/docker-compose.yml" ${params}
  )
}

# Pull the images from a docker compose file, if they're not already cached.
# This avoid slow remote calls from `docker-compose pull` which delegates
# to `docker pull` which always hits the remote docker repo, even if the image
# is already cached.
function dind::docker_compose_lazy_pull {
  for img in $(grep '^\s*image:\s' "${KUBE_ROOT}/hack/dind/docker-compose.yml" | sed 's/[ \t]*image:[ \t]*//'); do
    read repo tag <<<$(echo "${img} "| sed 's/:/ /')
    if [[ "${repo}" = k8s.io/kubernetes-dind* ]]; then
      continue
    fi
    if [ -z "${tag}" ]; then
      tag="latest"
    fi
    if ! docker images "${repo}" | awk '{print $2;}' | grep -q "${tag}"; then
      docker pull "${img}"
    fi
  done
}

# Generate kubeconfig data for the created cluster.
function dind::create-kubeconfig {
  local -r auth_dir="${DOCKER_IN_DOCKER_WORK_DIR}/auth"
  local kubectl="${KUBE_ROOT}/cluster/kubectl.sh"

  local token="$(cut -d, -f1 ${auth_dir}/token-users)"
  "${kubectl}" config set-cluster "dind" --server="${KUBE_SERVER}" --certificate-authority="${auth_dir}/ca.pem"
  "${kubectl}" config set-context "dind" --cluster="dind" --user="cluster-admin"
  "${kubectl}" config set-credentials cluster-admin --token="${token}"
  "${kubectl}" config use-context "dind" --cluster="dind"

   echo "Wrote config for dind context" 1>&2
}

# Must ensure that the following ENV vars are set
function dind::detect-master {
  KUBE_MASTER_IP="${APISERVER_ADDRESS}:6443"
  KUBE_SERVER="https://${KUBE_MASTER_IP}"

  echo "KUBE_MASTER_IP: $KUBE_MASTER_IP" 1>&2
}

# Get minion IP addresses and store in KUBE_NODE_IP_ADDRESSES[]
function dind::detect-nodes {
  local docker_ids=$(docker ps --filter="name=dind_node" --quiet)
  if [ -z "${docker_ids}" ]; then
    echo "ERROR: node(s) not running" 1>&2
    return 1
  fi
  while read -r docker_id; do
    local minion_ip=$(docker inspect --format="{{.NetworkSettings.IPAddress}}" "${docker_id}")
    KUBE_NODE_IP_ADDRESSES+=("${minion_ip}")
  done <<< "$docker_ids"
  echo "KUBE_NODE_IP_ADDRESSES: [${KUBE_NODE_IP_ADDRESSES[*]}]" 1>&2
}

# Verify prereqs on host machine
function dind::verify-prereqs {
  dind::step "Verifying required commands"
  hash docker 2>/dev/null || { echo "Missing required command: docker" 1>&2; exit 1; }
  hash docker 2>/dev/null || { echo "Missing required command: docker-compose" 1>&2; exit 1; }
  docker run busybox grep -q -w -e "overlay\|aufs" /proc/filesystems || {
    echo "Missing required kernel filesystem support: overlay or aufs."
    echo "Run 'sudo modprobe overlay' or 'sudo modprobe aufs' (on Ubuntu) and try again."
    exit 1
  }
}

# Initialize
function dind::init_auth {
  local -r auth_dir="${DOCKER_IN_DOCKER_WORK_DIR}/auth"

  dind::step "Creating auth directory:" "${auth_dir}"
  mkdir -p "${auth_dir}"
  ! selinuxenabled 2>&1 || sudo chcon -Rt svirt_sandbox_file_t -l s0 "${auth_dir}"
  rm -rf "${auth_dir}"/*

  dind::step "Creating service accounts key:" "${auth_dir}/service-accounts-key.pem"
  openssl genrsa -out "${auth_dir}/service-accounts-key.pem" 2048 &>/dev/null

  local -r BASIC_PASSWORD="$(openssl rand -hex 16)"
  local -r KUBELET_TOKEN="$(openssl rand -hex 32)"
  echo "${BASIC_PASSWORD},admin,admin" > ${auth_dir}/basic-users
  echo "${KUBELET_TOKEN},kubelet,kubelet" > ${auth_dir}/token-users
  dind::step "Creating credentials:" "admin:${BASIC_PASSWORD}, kubelet token"

  dind::step "Create TLS certs & keys:"
  docker run -i  --entrypoint /bin/bash -v "${auth_dir}:/certs" -w /certs cfssl/cfssl:latest -ec "$(cat <<EOF
    cd /certs
    echo '{"CN":"CA","key":{"algo":"rsa","size":2048}}' | cfssl gencert -initca - | cfssljson -bare ca -
    echo '{"signing":{"default":{"expiry":"43800h","usages":["signing","key encipherment","server auth","client auth"]}}}' > ca-config.json
    echo '{"CN":"'apiserver'","hosts":[""],"key":{"algo":"rsa","size":2048}}' | \
      cfssl gencert -ca=ca.pem -ca-key=ca-key.pem -config=ca-config.json -hostname=apiserver,kubernetes,kubernetes.default.svc.${DNS_DOMAIN},${APISERVER_SERVICE_IP},${APISERVER_ADDRESS} - | \
      cfssljson -bare apiserver
EOF
  )"
}

# Instantiate a kubernetes cluster.
function dind::kube-up {
  # Pull before `docker-compose up` to avoid timeouts caused by slow pulls during deployment.
  dind::step "Pulling docker images"
  dind::docker_compose_lazy_pull

  if [ "${DOCKER_IN_DOCKER_SKIP_BUILD}" != "true" ]; then
    dind::step "Building docker images"
    "${KUBE_ROOT}/hack/dind/image/build.sh"
  fi

  dind::init_auth

  dind::step "Starting dind cluster"
  dind::docker_compose up -d --force-recreate
  dind::step "Scaling dind cluster to ${NUM_NODES} slaves"
  dind::docker_compose scale node=${NUM_NODES}

  dind::step -n "Waiting for https://${APISERVER_ADDRESS}:6443 to be healthy"
  while ! curl -o /dev/null -s --cacert ${DOCKER_IN_DOCKER_WORK_DIR}/auth/ca.pem https://${APISERVER_ADDRESS}:6443; do
    sleep 1
    echo -n "."
  done
  echo

  dind::detect-master
  dind::detect-nodes
  dind::create-kubeconfig

  if [ "${ENABLE_CLUSTER_DNS}" == "true" ]; then
    dind::deploy-dns
  fi
  if [ "${ENABLE_CLUSTER_UI}" == "true" ]; then
    dind::deploy-ui
  fi

  # Wait for addons to deploy
  dind::await_ready "kube-dns" "${DOCKER_IN_DOCKER_ADDON_TIMEOUT}"
  dind::await_ready "kubernetes-dashboard" "${DOCKER_IN_DOCKER_ADDON_TIMEOUT}"
}

function dind::deploy-dns {
  dind::step "Deploying kube-dns"
  "${KUBE_ROOT}/cluster/kubectl.sh" create -f <(
    for f in skydns-rc.yaml skydns-svc.yaml; do
      echo "---"
      eval "cat <<EOF
$(<"${KUBE_ROOT}/cluster/addons/dns/${f}.sed")
EOF
" 2>/dev/null
    done
  )
}

function dind::deploy-ui {
  dind::step "Deploying dashboard"
  "${KUBE_ROOT}/cluster/kubectl.sh" create -f "${KUBE_ROOT}/cluster/addons/dashboard/dashboard-controller.yaml"
  "${KUBE_ROOT}/cluster/kubectl.sh" create -f "${KUBE_ROOT}/cluster/addons/dashboard/dashboard-service.yaml"
}

function dind::validate-cluster {
  dind::step "Validating dind cluster"

  # Do not validate cluster size. There will be zero k8s minions until a pod is created.
  # TODO(karlkfi): use componentstatuses or equivalent when it supports non-localhost core components

  # Validate immediate cluster reachability and responsiveness
  echo "KubeDNS: $(dind::addon_status 'kube-dns')"
  echo "Kubernetes Dashboard: $(dind::addon_status 'kubernetes-dashboard')"
}

# Delete a kubernetes cluster
function dind::kube-down {
  dind::step "Stopping dind cluster"
  # Since restoring a stopped cluster is not yet supported, use the nuclear option
  dind::docker_compose kill
  dind::docker_compose rm -f --all
}

# Waits for a kube-system pod (of the provided name) to have the phase/status "Running".
function dind::await_ready {
  local pod_name="$1"
  local max_attempts="$2"
  local phase="Unknown"
  echo -n "${pod_name}: "
  local n=0
  until [ ${n} -ge ${max_attempts} ]; do
    phase=$(dind::addon_status "${pod_name}")
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
function dind::addon_status {
  local pod_name="$1"
  local kubectl="${KUBE_ROOT}/cluster/kubectl.sh"
  local phase=$("${kubectl}" get pods --namespace=kube-system -l k8s-app=${pod_name} -o template --template="{{(index .items 0).status.phase}}" 2>/dev/null)
  phase="${phase:-Unknown}"
  echo "${phase}"
}

function dind::step {
  local OPTS=""
  if [ "$1" = "-n" ]; then
    shift
    OPTS+="-n"
  fi
  GREEN="${1}"
  shift
  echo -e ${OPTS} "\x1B[97m* \x1B[92m${GREEN}\x1B[39m $*" 1>&2
}

if [ $(basename "$0") = dind-up-cluster.sh ]; then
    source "${KUBE_ROOT}/hack/dind/config.sh"
    dind::kube-up
    if [ "${1:-}" = "-w" ]; then
      trap "echo; dind::kube-down" INT
      echo
      echo "Press Ctrl-C to shutdown cluster"
      while true; do sleep 1; done
    fi
elif [ $(basename "$0") = dind-down-cluster.sh ]; then
  source "${KUBE_ROOT}/hack/dind/config.sh"
  dind::kube-down
fi