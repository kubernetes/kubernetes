#!/bin/bash

# Copyright 2015 The Kubernetes Authors.
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

# Common utilites for kube-up/kube-down

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..

DEFAULT_KUBECONFIG="${HOME}/.kube/config"

source "${KUBE_ROOT}/cluster/lib/util.sh"
source "${KUBE_ROOT}/cluster/lib/logging.sh"
# KUBE_RELEASE_VERSION_REGEX matches things like "v1.2.3" or "v1.2.3-alpha.4"
#
# NOTE This must match the version_regex in build/common.sh
# kube::release::parse_and_validate_release_version()
KUBE_RELEASE_VERSION_REGEX="^v(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)(-(beta|alpha)\\.(0|[1-9][0-9]*))?$"
KUBE_RELEASE_VERSION_DASHED_REGEX="v(0|[1-9][0-9]*)-(0|[1-9][0-9]*)-(0|[1-9][0-9]*)(-(beta|alpha)-(0|[1-9][0-9]*))?"

# KUBE_CI_VERSION_REGEX matches things like "v1.2.3-alpha.4.56+abcdefg" This
#
# NOTE This must match the version_regex in build/common.sh
# kube::release::parse_and_validate_ci_version()
KUBE_CI_VERSION_REGEX="^v(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)-(beta|alpha)\\.(0|[1-9][0-9]*)(\\.(0|[1-9][0-9]*)\\+[-0-9a-z]*)?$"
KUBE_CI_VERSION_DASHED_REGEX="^v(0|[1-9][0-9]*)-(0|[1-9][0-9]*)-(0|[1-9][0-9]*)-(beta|alpha)-(0|[1-9][0-9]*)(-(0|[1-9][0-9]*)\\+[-0-9a-z]*)?"

# Generate kubeconfig data for the created cluster.
# Assumed vars:
#   KUBE_USER
#   KUBE_PASSWORD
#   KUBE_MASTER_IP
#   KUBECONFIG
#   CONTEXT
#
# If the apiserver supports bearer auth, also provide:
#   KUBE_BEARER_TOKEN
#
# If the kubeconfig context being created should NOT be set as the current context
# SECONDARY_KUBECONFIG=true
#
# To explicitly name the context being created, use OVERRIDE_CONTEXT
#
# The following can be omitted for --insecure-skip-tls-verify
#   KUBE_CERT
#   KUBE_KEY
#   CA_CERT
function create-kubeconfig() {
  KUBECONFIG=${KUBECONFIG:-$DEFAULT_KUBECONFIG}
  local kubectl="${KUBE_ROOT}/cluster/kubectl.sh"
  SECONDARY_KUBECONFIG=${SECONDARY_KUBECONFIG:-}
  OVERRIDE_CONTEXT=${OVERRIDE_CONTEXT:-}

  if [[ "$OVERRIDE_CONTEXT" != "" ]];then
      CONTEXT=$OVERRIDE_CONTEXT
  fi

  # KUBECONFIG determines the file we write to, but it may not exist yet
  if [[ ! -e "${KUBECONFIG}" ]]; then
    mkdir -p $(dirname "${KUBECONFIG}")
    touch "${KUBECONFIG}"
  fi
  local cluster_args=(
      "--server=${KUBE_SERVER:-https://${KUBE_MASTER_IP}}"
  )
  if [[ -z "${CA_CERT:-}" ]]; then
    cluster_args+=("--insecure-skip-tls-verify=true")
  else
    cluster_args+=(
      "--certificate-authority=${CA_CERT}"
      "--embed-certs=true"
    )
  fi

  local user_args=()
  if [[ ! -z "${KUBE_BEARER_TOKEN:-}" ]]; then
    user_args+=(
     "--token=${KUBE_BEARER_TOKEN}"
    )
  elif [[ ! -z "${KUBE_USER:-}" && ! -z "${KUBE_PASSWORD:-}" ]]; then
    user_args+=(
     "--username=${KUBE_USER}"
     "--password=${KUBE_PASSWORD}"
    )
  fi
  if [[ ! -z "${KUBE_CERT:-}" && ! -z "${KUBE_KEY:-}" ]]; then
    user_args+=(
     "--client-certificate=${KUBE_CERT}"
     "--client-key=${KUBE_KEY}"
     "--embed-certs=true"
    )
  fi

  KUBECONFIG="${KUBECONFIG}" "${kubectl}" config set-cluster "${CONTEXT}" "${cluster_args[@]}"
  if [[ -n "${user_args[@]:-}" ]]; then
    KUBECONFIG="${KUBECONFIG}" "${kubectl}" config set-credentials "${CONTEXT}" "${user_args[@]}"
  fi
  KUBECONFIG="${KUBECONFIG}" "${kubectl}" config set-context "${CONTEXT}" --cluster="${CONTEXT}" --user="${CONTEXT}"

  if [[ "${SECONDARY_KUBECONFIG}" != "true" ]];then
      KUBECONFIG="${KUBECONFIG}" "${kubectl}" config use-context "${CONTEXT}"  --cluster="${CONTEXT}"
  fi

  # If we have a bearer token, also create a credential entry with basic auth
  # so that it is easy to discover the basic auth password for your cluster
  # to use in a web browser.
  if [[ ! -z "${KUBE_BEARER_TOKEN:-}" && ! -z "${KUBE_USER:-}" && ! -z "${KUBE_PASSWORD:-}" ]]; then
    KUBECONFIG="${KUBECONFIG}" "${kubectl}" config set-credentials "${CONTEXT}-basic-auth" "--username=${KUBE_USER}" "--password=${KUBE_PASSWORD}"
  fi

   echo "Wrote config for ${CONTEXT} to ${KUBECONFIG}"
}

# Clear kubeconfig data for a context
# Assumed vars:
#   KUBECONFIG
#   CONTEXT
#
# To explicitly name the context being removed, use OVERRIDE_CONTEXT
function clear-kubeconfig() {
  export KUBECONFIG=${KUBECONFIG:-$DEFAULT_KUBECONFIG}
  OVERRIDE_CONTEXT=${OVERRIDE_CONTEXT:-}

  if [[ "$OVERRIDE_CONTEXT" != "" ]];then
      CONTEXT=$OVERRIDE_CONTEXT
  fi

  local kubectl="${KUBE_ROOT}/cluster/kubectl.sh"
  "${kubectl}" config unset "clusters.${CONTEXT}"
  "${kubectl}" config unset "users.${CONTEXT}"
  "${kubectl}" config unset "users.${CONTEXT}-basic-auth"
  "${kubectl}" config unset "contexts.${CONTEXT}"

  local cc=$("${kubectl}" config view -o jsonpath='{.current-context}')
  if [[ "${cc}" == "${CONTEXT}" ]]; then
    "${kubectl}" config unset current-context
  fi

  echo "Cleared config for ${CONTEXT} from ${KUBECONFIG}"
}

# Creates a kubeconfig file with the credentials for only the current-context
# cluster. This is used by federation to create secrets in test setup.
function create-kubeconfig-for-federation() {
  if [[ "${FEDERATION:-}" == "true" ]]; then
    echo "creating kubeconfig for federation secret"
    local kubectl="${KUBE_ROOT}/cluster/kubectl.sh"
    local cc=$("${kubectl}" config view -o jsonpath='{.current-context}')
    KUBECONFIG_DIR=$(dirname ${KUBECONFIG:-$DEFAULT_KUBECONFIG})
    KUBECONFIG_PATH="${KUBECONFIG_DIR}/federation/kubernetes-apiserver/${cc}"
    mkdir -p "${KUBECONFIG_PATH}"
    "${kubectl}" config view --minify --flatten > "${KUBECONFIG_PATH}/kubeconfig"
  fi
}

function tear_down_alive_resources() {
  local kubectl="${KUBE_ROOT}/cluster/kubectl.sh"
  "${kubectl}" delete rc --all || true
  "${kubectl}" delete pods --all || true
  "${kubectl}" delete svc --all || true
  "${kubectl}" delete pvc --all || true
}

# Gets username, password for the current-context in kubeconfig, if they exist.
# Assumed vars:
#   KUBECONFIG  # if unset, defaults to global
#   KUBE_CONTEXT  # if unset, defaults to current-context
#
# Vars set:
#   KUBE_USER
#   KUBE_PASSWORD
#
# KUBE_USER,KUBE_PASSWORD will be empty if no current-context is set, or
# the current-context user does not exist or contain basicauth entries.
function get-kubeconfig-basicauth() {
  export KUBECONFIG=${KUBECONFIG:-$DEFAULT_KUBECONFIG}

  local cc=$("${KUBE_ROOT}/cluster/kubectl.sh" config view -o jsonpath="{.current-context}")
  if [[ ! -z "${KUBE_CONTEXT:-}" ]]; then
    cc="${KUBE_CONTEXT}"
  fi
  local user=$("${KUBE_ROOT}/cluster/kubectl.sh" config view -o jsonpath="{.contexts[?(@.name == \"${cc}\")].context.user}")
  get-kubeconfig-user-basicauth "${user}"

  if [[ -z "${KUBE_USER:-}" || -z "${KUBE_PASSWORD:-}" ]]; then
    # kube-up stores username/password in a an additional kubeconfig section
    # suffixed with "-basic-auth". Cloudproviders like GKE store in directly
    # in the top level section along with the other credential information.
    # TODO: Handle this uniformly, either get rid of "basic-auth" or
    # consolidate its usage into a function across scripts in cluster/
    get-kubeconfig-user-basicauth "${user}-basic-auth"
  fi
}

# Sets KUBE_USER and KUBE_PASSWORD to the username and password specified in
# the kubeconfig section corresponding to $1.
#
# Args:
#   $1 kubeconfig section to look for basic auth (eg: user or user-basic-auth).
# Assumed vars:
#   KUBE_ROOT
# Vars set:
#   KUBE_USER
#   KUBE_PASSWORD
function get-kubeconfig-user-basicauth() {
  KUBE_USER=$("${KUBE_ROOT}/cluster/kubectl.sh" config view -o jsonpath="{.users[?(@.name == \"$1\")].user.username}")
  KUBE_PASSWORD=$("${KUBE_ROOT}/cluster/kubectl.sh" config view -o jsonpath="{.users[?(@.name == \"$1\")].user.password}")
}

# Generate basic auth user and password.

# Vars set:
#   KUBE_USER
#   KUBE_PASSWORD
function gen-kube-basicauth() {
    KUBE_USER=admin
    KUBE_PASSWORD=$(python -c 'import string,random; print("".join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(16)))')
}

# Get the bearer token for the current-context in kubeconfig if one exists.
# Assumed vars:
#   KUBECONFIG  # if unset, defaults to global
#   KUBE_CONTEXT  # if unset, defaults to current-context
#
# Vars set:
#   KUBE_BEARER_TOKEN
#
# KUBE_BEARER_TOKEN will be empty if no current-context is set, or the
# current-context user does not exist or contain a bearer token entry.
function get-kubeconfig-bearertoken() {
  export KUBECONFIG=${KUBECONFIG:-$DEFAULT_KUBECONFIG}

  local cc=$("${KUBE_ROOT}/cluster/kubectl.sh" config view -o jsonpath="{.current-context}")
  if [[ ! -z "${KUBE_CONTEXT:-}" ]]; then
    cc="${KUBE_CONTEXT}"
  fi
  local user=$("${KUBE_ROOT}/cluster/kubectl.sh" config view -o jsonpath="{.contexts[?(@.name == \"${cc}\")].context.user}")
  KUBE_BEARER_TOKEN=$("${KUBE_ROOT}/cluster/kubectl.sh" config view -o jsonpath="{.users[?(@.name == \"${user}\")].user.token}")
}

# Generate bearer token.
#
# Vars set:
#   KUBE_BEARER_TOKEN
function gen-kube-bearertoken() {
    KUBE_BEARER_TOKEN=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
}

# Generate uid
# This function only works on systems with python. It generates a time based
# UID instead of a UUID because GCE has a name length limit.
#
# Vars set:
#   KUBE_UID
function gen-uid {
    KUBE_UID=$(python -c 'import uuid; print(uuid.uuid1().fields[0])')
}


function load-or-gen-kube-basicauth() {
  if [[ ! -z "${KUBE_CONTEXT:-}" ]]; then
    get-kubeconfig-basicauth
  fi

  if [[ -z "${KUBE_USER:-}" || -z "${KUBE_PASSWORD:-}" ]]; then
    gen-kube-basicauth
  fi

  # Make sure they don't contain any funny characters.
  if ! [[ "${KUBE_USER}" =~ ^[-._@a-zA-Z0-9]+$ ]]; then
    echo "Bad KUBE_USER string."
    exit 1
  fi
  if ! [[ "${KUBE_PASSWORD}" =~ ^[-._@#%/a-zA-Z0-9]+$ ]]; then
    echo "Bad KUBE_PASSWORD string."
    exit 1
  fi
}

function load-or-gen-kube-bearertoken() {
  if [[ ! -z "${KUBE_CONTEXT:-}" ]]; then
    get-kubeconfig-bearertoken
  fi
  if [[ -z "${KUBE_BEARER_TOKEN:-}" ]]; then
    gen-kube-bearertoken
  fi
}

# Get the master IP for the current-context in kubeconfig if one exists.
#
# Assumed vars:
#   KUBECONFIG  # if unset, defaults to global
#   KUBE_CONTEXT  # if unset, defaults to current-context
#
# Vars set:
#   KUBE_MASTER_URL
#
# KUBE_MASTER_URL will be empty if no current-context is set, or the
# current-context user does not exist or contain a server entry.
function detect-master-from-kubeconfig() {
  export KUBECONFIG=${KUBECONFIG:-$DEFAULT_KUBECONFIG}

  local cc=$("${KUBE_ROOT}/cluster/kubectl.sh" config view -o jsonpath="{.current-context}")
  if [[ ! -z "${KUBE_CONTEXT:-}" ]]; then
    cc="${KUBE_CONTEXT}"
  fi
  local cluster=$("${KUBE_ROOT}/cluster/kubectl.sh" config view -o jsonpath="{.contexts[?(@.name == \"${cc}\")].context.cluster}")
  KUBE_MASTER_URL=$("${KUBE_ROOT}/cluster/kubectl.sh" config view -o jsonpath="{.clusters[?(@.name == \"${cluster}\")].cluster.server}")
}

# Sets KUBE_VERSION variable to the proper version number (e.g. "v1.0.6",
# "v1.2.0-alpha.1.881+376438b69c7612") or a version' publication of the form
# <path>/<version> (e.g. "release/stable",' "ci/latest-1").
#
# See the docs on getting builds for more information about version
# publication.
#
# Args:
#   $1 version string from command line
# Vars set:
#   KUBE_VERSION
function set_binary_version() {
  if [[ "${1}" =~ "/" ]]; then
    IFS='/' read -a path <<< "${1}"
    if [[ "${path[0]}" == "release" ]]; then
      KUBE_VERSION=$(gsutil cat "gs://kubernetes-release/${1}.txt")
    else
      KUBE_VERSION=$(gsutil cat "gs://kubernetes-release-dev/${1}.txt")
    fi
  else
    KUBE_VERSION=${1}
  fi
}

# Figure out which binary use on the server and assure it is available.
# If KUBE_VERSION is specified use binaries specified by it, otherwise
# use local dev binaries.
#
# Assumed vars:
#   KUBE_VERSION
#   KUBE_RELEASE_VERSION_REGEX
#   KUBE_CI_VERSION_REGEX
# Vars set:
#   KUBE_TAR_HASH
#   SERVER_BINARY_TAR_URL
#   SERVER_BINARY_TAR_HASH
#   SALT_TAR_URL
#   SALT_TAR_HASH
function tars_from_version() {
  if [[ -z "${KUBE_VERSION-}" ]]; then
    find-release-tars
    upload-server-tars
  elif [[ ${KUBE_VERSION} =~ ${KUBE_RELEASE_VERSION_REGEX} ]]; then
    SERVER_BINARY_TAR_URL="https://storage.googleapis.com/kubernetes-release/release/${KUBE_VERSION}/kubernetes-server-linux-amd64.tar.gz"
    SALT_TAR_URL="https://storage.googleapis.com/kubernetes-release/release/${KUBE_VERSION}/kubernetes-salt.tar.gz"
    # TODO: Clean this up.
    KUBE_MANIFESTS_TAR_URL="${SERVER_BINARY_TAR_URL/server-linux-amd64/manifests}"
    KUBE_MANIFESTS_TAR_HASH=$(curl ${KUBE_MANIFESTS_TAR_URL} 2>/dev/null | sha1sum | awk '{print $1}')
  elif [[ ${KUBE_VERSION} =~ ${KUBE_CI_VERSION_REGEX} ]]; then
    SERVER_BINARY_TAR_URL="https://storage.googleapis.com/kubernetes-release-dev/ci/${KUBE_VERSION}/kubernetes-server-linux-amd64.tar.gz"
    SALT_TAR_URL="https://storage.googleapis.com/kubernetes-release-dev/ci/${KUBE_VERSION}/kubernetes-salt.tar.gz"
    # TODO: Clean this up.
    KUBE_MANIFESTS_TAR_URL="${SERVER_BINARY_TAR_URL/server-linux-amd64/manifests}"
    KUBE_MANIFESTS_TAR_HASH=$(curl ${KUBE_MANIFESTS_TAR_URL} 2>/dev/null | sha1sum | awk '{print $1}')
  else
    echo "Version doesn't match regexp" >&2
    exit 1
  fi
  if ! SERVER_BINARY_TAR_HASH=$(curl -Ss --fail "${SERVER_BINARY_TAR_URL}.sha1"); then
    echo "Failure trying to curl release .sha1"
  fi
  if ! SALT_TAR_HASH=$(curl -Ss --fail "${SALT_TAR_URL}.sha1"); then
    echo "Failure trying to curl Salt tar .sha1"
  fi

  if ! curl -Ss --head "${SERVER_BINARY_TAR_URL}" >&/dev/null; then
    echo "Can't find release at ${SERVER_BINARY_TAR_URL}" >&2
    exit 1
  fi
  if ! curl -Ss --head "${SALT_TAR_URL}" >&/dev/null; then
    echo "Can't find Salt tar at ${SALT_TAR_URL}" >&2
    exit 1
  fi
}

# Verify and find the various tar files that we are going to use on the server.
#
# Assumed vars:
#   KUBE_ROOT
# Vars set:
#   SERVER_BINARY_TAR
#   SALT_TAR
#   KUBE_MANIFESTS_TAR
function find-release-tars() {
  SERVER_BINARY_TAR="${KUBE_ROOT}/server/kubernetes-server-linux-amd64.tar.gz"
  if [[ ! -f "${SERVER_BINARY_TAR}" ]]; then
    SERVER_BINARY_TAR="${KUBE_ROOT}/_output/release-tars/kubernetes-server-linux-amd64.tar.gz"
  fi
  if [[ ! -f "${SERVER_BINARY_TAR}" ]]; then
    echo "!!! Cannot find kubernetes-server-linux-amd64.tar.gz" >&2
    exit 1
  fi

  SALT_TAR="${KUBE_ROOT}/server/kubernetes-salt.tar.gz"
  if [[ ! -f "${SALT_TAR}" ]]; then
    SALT_TAR="${KUBE_ROOT}/_output/release-tars/kubernetes-salt.tar.gz"
  fi
  if [[ ! -f "${SALT_TAR}" ]]; then
    echo "!!! Cannot find kubernetes-salt.tar.gz" >&2
    exit 1
  fi

  # This tarball is used by GCI, Ubuntu Trusty, and CoreOS.
  KUBE_MANIFESTS_TAR=
  if [[ "${MASTER_OS_DISTRIBUTION:-}" == "trusty" || "${MASTER_OS_DISTRIBUTION:-}" == "gci" || "${MASTER_OS_DISTRIBUTION:-}" == "coreos" ]] || \
     [[ "${NODE_OS_DISTRIBUTION:-}" == "trusty" || "${NODE_OS_DISTRIBUTION:-}" == "gci" || "${NODE_OS_DISTRIBUTION:-}" == "coreos" ]] ; then
    KUBE_MANIFESTS_TAR="${KUBE_ROOT}/server/kubernetes-manifests.tar.gz"
    if [[ ! -f "${KUBE_MANIFESTS_TAR}" ]]; then
      KUBE_MANIFESTS_TAR="${KUBE_ROOT}/_output/release-tars/kubernetes-manifests.tar.gz"
    fi
    if [[ ! -f "${KUBE_MANIFESTS_TAR}" ]]; then
      echo "!!! Cannot find kubernetes-manifests.tar.gz" >&2
      exit 1
    fi
  fi
}

# Discover the git version of the current build package
#
# Assumed vars:
#   KUBE_ROOT
# Vars set:
#   KUBE_GIT_VERSION
function find-release-version() {
  KUBE_GIT_VERSION=""
  if [[ -f "${KUBE_ROOT}/version" ]]; then
    KUBE_GIT_VERSION="$(cat ${KUBE_ROOT}/version)"
  fi
  if [[ -f "${KUBE_ROOT}/_output/release-stage/full/kubernetes/version" ]]; then
    KUBE_GIT_VERSION="$(cat ${KUBE_ROOT}/_output/release-stage/full/kubernetes/version)"
  fi

  if [[ -z "${KUBE_GIT_VERSION}" ]]; then
    echo "!!! Cannot find release version"
    exit 1
  fi
}

function stage-images() {
  find-release-version
  find-release-tars

  KUBE_IMAGE_TAG="$(echo """${KUBE_GIT_VERSION}""" | sed 's/+/-/g')"

  local docker_wrapped_binaries=(
    "kube-apiserver"
    "kube-controller-manager"
    "kube-scheduler"
    "kube-proxy"
  )

  local docker_cmd=("docker")

  if [[ "${KUBE_DOCKER_REGISTRY}" == "gcr.io/"* ]]; then
    local docker_push_cmd=("gcloud" "docker")
  else
    local docker_push_cmd=("${docker_cmd[@]}")
  fi

  local temp_dir="$(mktemp -d -t 'kube-server-XXXX')"

  tar xzfv "${SERVER_BINARY_TAR}" -C "${temp_dir}" &> /dev/null

  for binary in "${docker_wrapped_binaries[@]}"; do
    local docker_tag="$(cat ${temp_dir}/kubernetes/server/bin/${binary}.docker_tag)"
    (
      "${docker_cmd[@]}" load -i "${temp_dir}/kubernetes/server/bin/${binary}.tar"
      "${docker_cmd[@]}" tag -f "gcr.io/google_containers/${binary}:${docker_tag}" "${KUBE_DOCKER_REGISTRY}/${binary}:${KUBE_IMAGE_TAG}"
      "${docker_push_cmd[@]}" push "${KUBE_DOCKER_REGISTRY}/${binary}:${KUBE_IMAGE_TAG}"
    ) &> "${temp_dir}/${binary}-push.log" &
  done

  kube::util::wait-for-jobs || {
    kube::log::error "unable to push images. see ${temp_dir}/*.log for more info."
    return 1
  }

  rm -rf "${temp_dir}"
  return 0
}

# Quote something appropriate for a yaml string.
#
# TODO(zmerlynn): Note that this function doesn't so much "quote" as
# "strip out quotes", and we really should be using a YAML library for
# this, but PyYAML isn't shipped by default, and *rant rant rant ... SIGH*
function yaml-quote {
  echo "'$(echo "${@:-}" | sed -e "s/'/''/g")'"
}

# Builds the RUNTIME_CONFIG var from other feature enable options (such as
# features in alpha)
function build-runtime-config() {
  # There is nothing to do here for now. Just using this function as a placeholder.
  :
}

# Writes the cluster name into a temporary file.
# Assumed vars
#   CLUSTER_NAME
function write-cluster-name {
  cat >"${KUBE_TEMP}/cluster-name.txt" << EOF
${CLUSTER_NAME}
EOF
}

function write-master-env {
  # If the user requested that the master be part of the cluster, set the
  # environment variable to program the master kubelet to register itself.
  if [[ "${REGISTER_MASTER_KUBELET:-}" == "true" ]]; then
    KUBELET_APISERVER="${MASTER_NAME}"
  fi

  build-kube-env true "${KUBE_TEMP}/master-kube-env.yaml"
}

function write-node-env {
  build-kube-env false "${KUBE_TEMP}/node-kube-env.yaml"
}

# $1: if 'true', we're building a master yaml, else a node
function build-kube-env {
  local master=$1
  local file=$2

  local server_binary_tar_url=$SERVER_BINARY_TAR_URL
  local salt_tar_url=$SALT_TAR_URL
  local kube_manifests_tar_url="${KUBE_MANIFESTS_TAR_URL:-}"
  if [[ "${master}" == "true" && "${MASTER_OS_DISTRIBUTION}" == "coreos" ]] || \
     [[ "${master}" == "false" && "${NODE_OS_DISTRIBUTION}" == "coreos" ]] ; then
    # TODO: Support fallback .tar.gz settings on CoreOS
    server_binary_tar_url=$(split_csv "${SERVER_BINARY_TAR_URL}")
    salt_tar_url=$(split_csv "${SALT_TAR_URL}")
    kube_manifests_tar_url=$(split_csv "${KUBE_MANIFESTS_TAR_URL}")
  fi

  build-runtime-config
  gen-uid

  rm -f ${file}
  cat >$file <<EOF
ENV_TIMESTAMP: $(yaml-quote $(date -u +%Y-%m-%dT%T%z))
INSTANCE_PREFIX: $(yaml-quote ${INSTANCE_PREFIX})
NODE_INSTANCE_PREFIX: $(yaml-quote ${NODE_INSTANCE_PREFIX})
NODE_TAGS: $(yaml-quote ${NODE_TAGS:-})
CLUSTER_IP_RANGE: $(yaml-quote ${CLUSTER_IP_RANGE:-10.244.0.0/16})
SERVER_BINARY_TAR_URL: $(yaml-quote ${server_binary_tar_url})
SERVER_BINARY_TAR_HASH: $(yaml-quote ${SERVER_BINARY_TAR_HASH})
SALT_TAR_URL: $(yaml-quote ${salt_tar_url})
SALT_TAR_HASH: $(yaml-quote ${SALT_TAR_HASH})
SERVICE_CLUSTER_IP_RANGE: $(yaml-quote ${SERVICE_CLUSTER_IP_RANGE})
KUBERNETES_MASTER_NAME: $(yaml-quote ${MASTER_NAME})
ALLOCATE_NODE_CIDRS: $(yaml-quote ${ALLOCATE_NODE_CIDRS:-false})
ENABLE_CLUSTER_MONITORING: $(yaml-quote ${ENABLE_CLUSTER_MONITORING:-none})
DOCKER_REGISTRY_MIRROR_URL: $(yaml-quote ${DOCKER_REGISTRY_MIRROR_URL:-})
ENABLE_L7_LOADBALANCING: $(yaml-quote ${ENABLE_L7_LOADBALANCING:-none})
ENABLE_CLUSTER_LOGGING: $(yaml-quote ${ENABLE_CLUSTER_LOGGING:-false})
ENABLE_CLUSTER_UI: $(yaml-quote ${ENABLE_CLUSTER_UI:-false})
ENABLE_NODE_PROBLEM_DETECTOR: $(yaml-quote ${ENABLE_NODE_PROBLEM_DETECTOR:-false})
ENABLE_NODE_LOGGING: $(yaml-quote ${ENABLE_NODE_LOGGING:-false})
ENABLE_RESCHEDULER: $(yaml-quote ${ENABLE_RESCHEDULER:-false})
LOGGING_DESTINATION: $(yaml-quote ${LOGGING_DESTINATION:-})
ELASTICSEARCH_LOGGING_REPLICAS: $(yaml-quote ${ELASTICSEARCH_LOGGING_REPLICAS:-})
ENABLE_CLUSTER_DNS: $(yaml-quote ${ENABLE_CLUSTER_DNS:-false})
ENABLE_CLUSTER_REGISTRY: $(yaml-quote ${ENABLE_CLUSTER_REGISTRY:-false})
CLUSTER_REGISTRY_DISK: $(yaml-quote ${CLUSTER_REGISTRY_DISK:-})
CLUSTER_REGISTRY_DISK_SIZE: $(yaml-quote ${CLUSTER_REGISTRY_DISK_SIZE:-})
DNS_REPLICAS: $(yaml-quote ${DNS_REPLICAS:-})
DNS_SERVER_IP: $(yaml-quote ${DNS_SERVER_IP:-})
DNS_DOMAIN: $(yaml-quote ${DNS_DOMAIN:-})
KUBELET_TOKEN: $(yaml-quote ${KUBELET_TOKEN:-})
KUBE_PROXY_TOKEN: $(yaml-quote ${KUBE_PROXY_TOKEN:-})
ADMISSION_CONTROL: $(yaml-quote ${ADMISSION_CONTROL:-})
MASTER_IP_RANGE: $(yaml-quote ${MASTER_IP_RANGE})
RUNTIME_CONFIG: $(yaml-quote ${RUNTIME_CONFIG})
CA_CERT: $(yaml-quote ${CA_CERT_BASE64:-})
KUBELET_CERT: $(yaml-quote ${KUBELET_CERT_BASE64:-})
KUBELET_KEY: $(yaml-quote ${KUBELET_KEY_BASE64:-})
NETWORK_PROVIDER: $(yaml-quote ${NETWORK_PROVIDER:-})
NETWORK_POLICY_PROVIDER: $(yaml-quote ${NETWORK_POLICY_PROVIDER:-})
PREPULL_E2E_IMAGES: $(yaml-quote ${PREPULL_E2E_IMAGES:-})
HAIRPIN_MODE: $(yaml-quote ${HAIRPIN_MODE:-})
OPENCONTRAIL_TAG: $(yaml-quote ${OPENCONTRAIL_TAG:-})
OPENCONTRAIL_KUBERNETES_TAG: $(yaml-quote ${OPENCONTRAIL_KUBERNETES_TAG:-})
OPENCONTRAIL_PUBLIC_SUBNET: $(yaml-quote ${OPENCONTRAIL_PUBLIC_SUBNET:-})
E2E_STORAGE_TEST_ENVIRONMENT: $(yaml-quote ${E2E_STORAGE_TEST_ENVIRONMENT:-})
KUBE_IMAGE_TAG: $(yaml-quote ${KUBE_IMAGE_TAG:-})
KUBE_DOCKER_REGISTRY: $(yaml-quote ${KUBE_DOCKER_REGISTRY:-})
KUBE_ADDON_REGISTRY: $(yaml-quote ${KUBE_ADDON_REGISTRY:-})
MULTIZONE: $(yaml-quote ${MULTIZONE:-})
NON_MASQUERADE_CIDR: $(yaml-quote ${NON_MASQUERADE_CIDR:-})
KUBE_UID: $(yaml-quote ${KUBE_UID:-})
EOF
  if [ -n "${KUBELET_PORT:-}" ]; then
    cat >>$file <<EOF
KUBELET_PORT: $(yaml-quote ${KUBELET_PORT})
EOF
  fi
  if [ -n "${KUBE_APISERVER_REQUEST_TIMEOUT:-}" ]; then
    cat >>$file <<EOF
KUBE_APISERVER_REQUEST_TIMEOUT: $(yaml-quote ${KUBE_APISERVER_REQUEST_TIMEOUT})
EOF
  fi
  if [ -n "${TERMINATED_POD_GC_THRESHOLD:-}" ]; then
    cat >>$file <<EOF
TERMINATED_POD_GC_THRESHOLD: $(yaml-quote ${TERMINATED_POD_GC_THRESHOLD})
EOF
  fi
  if [[ "${master}" == "true" && ("${MASTER_OS_DISTRIBUTION}" == "trusty" || "${MASTER_OS_DISTRIBUTION}" == "gci" || "${MASTER_OS_DISTRIBUTION}" == "coreos") ]] || \
     [[ "${master}" == "false" && ("${NODE_OS_DISTRIBUTION}" == "trusty" || "${NODE_OS_DISTRIBUTION}" == "gci" || "${NODE_OS_DISTRIBUTION}" == "coreos") ]] ; then
    cat >>$file <<EOF
KUBE_MANIFESTS_TAR_URL: $(yaml-quote ${kube_manifests_tar_url})
KUBE_MANIFESTS_TAR_HASH: $(yaml-quote ${KUBE_MANIFESTS_TAR_HASH})
EOF
  fi
  if [ -n "${TEST_CLUSTER:-}" ]; then
    cat >>$file <<EOF
TEST_CLUSTER: $(yaml-quote ${TEST_CLUSTER})
EOF
  fi
  if [ -n "${KUBELET_TEST_ARGS:-}" ]; then
      cat >>$file <<EOF
KUBELET_TEST_ARGS: $(yaml-quote ${KUBELET_TEST_ARGS})
EOF
  fi
  if [ -n "${KUBELET_TEST_LOG_LEVEL:-}" ]; then
      cat >>$file <<EOF
KUBELET_TEST_LOG_LEVEL: $(yaml-quote ${KUBELET_TEST_LOG_LEVEL})
EOF
  fi
  if [ -n "${DOCKER_TEST_LOG_LEVEL:-}" ]; then
      cat >>$file <<EOF
DOCKER_TEST_LOG_LEVEL: $(yaml-quote ${DOCKER_TEST_LOG_LEVEL})
EOF
  fi
  if [ -n "${ENABLE_CUSTOM_METRICS:-}" ]; then
    cat >>$file <<EOF
ENABLE_CUSTOM_METRICS: $(yaml-quote ${ENABLE_CUSTOM_METRICS})
EOF
  fi
  if [ -n "${FEATURE_GATES:-}" ]; then
    cat >>$file <<EOF
FEATURE_GATES: $(yaml-quote ${FEATURE_GATES})
EOF
  fi
  if [[ "${master}" == "true" ]]; then
    # Master-only env vars.
    cat >>$file <<EOF
KUBERNETES_MASTER: $(yaml-quote "true")
KUBE_USER: $(yaml-quote ${KUBE_USER})
KUBE_PASSWORD: $(yaml-quote ${KUBE_PASSWORD})
KUBE_BEARER_TOKEN: $(yaml-quote ${KUBE_BEARER_TOKEN})
MASTER_CERT: $(yaml-quote ${MASTER_CERT_BASE64:-})
MASTER_KEY: $(yaml-quote ${MASTER_KEY_BASE64:-})
KUBECFG_CERT: $(yaml-quote ${KUBECFG_CERT_BASE64:-})
KUBECFG_KEY: $(yaml-quote ${KUBECFG_KEY_BASE64:-})
KUBELET_APISERVER: $(yaml-quote ${KUBELET_APISERVER:-})
ENABLE_MANIFEST_URL: $(yaml-quote ${ENABLE_MANIFEST_URL:-false})
MANIFEST_URL: $(yaml-quote ${MANIFEST_URL:-})
MANIFEST_URL_HEADER: $(yaml-quote ${MANIFEST_URL_HEADER:-})
NUM_NODES: $(yaml-quote ${NUM_NODES})
STORAGE_BACKEND: $(yaml-quote ${STORAGE_BACKEND:-})
ENABLE_GARBAGE_COLLECTOR: $(yaml-quote ${ENABLE_GARBAGE_COLLECTOR:-})
EOF
    if [ -n "${TEST_ETCD_VERSION:-}" ]; then
      cat >>$file <<EOF
TEST_ETCD_VERSION: $(yaml-quote ${TEST_ETCD_VERSION})
EOF
    fi
    if [ -n "${APISERVER_TEST_ARGS:-}" ]; then
      cat >>$file <<EOF
APISERVER_TEST_ARGS: $(yaml-quote ${APISERVER_TEST_ARGS})
EOF
    fi
    if [ -n "${APISERVER_TEST_LOG_LEVEL:-}" ]; then
      cat >>$file <<EOF
APISERVER_TEST_LOG_LEVEL: $(yaml-quote ${APISERVER_TEST_LOG_LEVEL})
EOF
    fi
    if [ -n "${CONTROLLER_MANAGER_TEST_ARGS:-}" ]; then
      cat >>$file <<EOF
CONTROLLER_MANAGER_TEST_ARGS: $(yaml-quote ${CONTROLLER_MANAGER_TEST_ARGS})
EOF
    fi
    if [ -n "${CONTROLLER_MANAGER_TEST_LOG_LEVEL:-}" ]; then
      cat >>$file <<EOF
CONTROLLER_MANAGER_TEST_LOG_LEVEL: $(yaml-quote ${CONTROLLER_MANAGER_TEST_LOG_LEVEL})
EOF
    fi
    if [ -n "${SCHEDULER_TEST_ARGS:-}" ]; then
      cat >>$file <<EOF
SCHEDULER_TEST_ARGS: $(yaml-quote ${SCHEDULER_TEST_ARGS})
EOF
    fi
    if [ -n "${SCHEDULER_TEST_LOG_LEVEL:-}" ]; then
      cat >>$file <<EOF
SCHEDULER_TEST_LOG_LEVEL: $(yaml-quote ${SCHEDULER_TEST_LOG_LEVEL})
EOF
    fi
    if [ -n "${INITIAL_ETCD_CLUSTER:-}" ]; then
      cat >>$file <<EOF
INITIAL_ETCD_CLUSTER: $(yaml-quote ${INITIAL_ETCD_CLUSTER})
EOF
    fi

  else
    # Node-only env vars.
    cat >>$file <<EOF
KUBERNETES_MASTER: $(yaml-quote "false")
ZONE: $(yaml-quote ${ZONE})
EXTRA_DOCKER_OPTS: $(yaml-quote ${EXTRA_DOCKER_OPTS:-})
EOF
    if [ -n "${KUBEPROXY_TEST_ARGS:-}" ]; then
      cat >>$file <<EOF
KUBEPROXY_TEST_ARGS: $(yaml-quote ${KUBEPROXY_TEST_ARGS})
EOF
    fi
    if [ -n "${KUBEPROXY_TEST_LOG_LEVEL:-}" ]; then
      cat >>$file <<EOF
KUBEPROXY_TEST_LOG_LEVEL: $(yaml-quote ${KUBEPROXY_TEST_LOG_LEVEL})
EOF
    fi
  fi
  if [ -n "${NODE_LABELS:-}" ]; then
      cat >>$file <<EOF
NODE_LABELS: $(yaml-quote ${NODE_LABELS})
EOF
    fi
  if [ -n "${EVICTION_HARD:-}" ]; then
      cat >>$file <<EOF
EVICTION_HARD: $(yaml-quote ${EVICTION_HARD})
EOF
    fi
  if [[ "${master}" == "true" && "${MASTER_OS_DISTRIBUTION}" == "coreos" ]] || \
     [[ "${master}" == "false" && "${NODE_OS_DISTRIBUTION}" == "coreos" ]]; then
    # CoreOS-only env vars. TODO(yifan): Make them available on other distros.
    cat >>$file <<EOF
KUBERNETES_CONTAINER_RUNTIME: $(yaml-quote ${CONTAINER_RUNTIME:-rkt})
RKT_VERSION: $(yaml-quote ${RKT_VERSION:-})
RKT_PATH: $(yaml-quote ${RKT_PATH:-})
RKT_STAGE1_IMAGE: $(yaml-quote ${RKT_STAGE1_IMAGE:-})
KUBERNETES_CONFIGURE_CBR0: $(yaml-quote ${KUBERNETES_CONFIGURE_CBR0:-true})
EOF
  fi
  if [[ "${ENABLE_CLUSTER_AUTOSCALER}" == "true" ]]; then
      cat >>$file <<EOF
ENABLE_CLUSTER_AUTOSCALER: $(yaml-quote ${ENABLE_CLUSTER_AUTOSCALER})
AUTOSCALER_MIG_CONFIG: $(yaml-quote ${AUTOSCALER_MIG_CONFIG})
EOF
  fi

  # Federation specific environment variables.
  if [[ -n "${FEDERATION:-}" ]]; then
    cat >>$file <<EOF
FEDERATION: $(yaml-quote ${FEDERATION})
EOF
  fi
  if [ -n "${FEDERATIONS_DOMAIN_MAP:-}" ]; then
    cat >>$file <<EOF
FEDERATIONS_DOMAIN_MAP: $(yaml-quote ${FEDERATIONS_DOMAIN_MAP})
EOF
  fi
  if [ -n "${FEDERATION_NAME:-}" ]; then
    cat >>$file <<EOF
FEDERATION_NAME: $(yaml-quote ${FEDERATION_NAME})
EOF
  fi
  if [ -n "${DNS_ZONE_NAME:-}" ]; then
    cat >>$file <<EOF
DNS_ZONE_NAME: $(yaml-quote ${DNS_ZONE_NAME})
EOF
  fi
  if [ -n "${SCHEDULING_ALGORITHM_PROVIDER:-}" ]; then
    cat >>$file <<EOF
SCHEDULING_ALGORITHM_PROVIDER: $(yaml-quote ${SCHEDULING_ALGORITHM_PROVIDER})
EOF
  fi
}

function sha1sum-file() {
  if which shasum >/dev/null 2>&1; then
    shasum -a1 "$1" | awk '{ print $1 }'
  else
    sha1sum "$1" | awk '{ print $1 }'
  fi
}

# Create certificate pairs for the cluster.
# $1: The public IP for the master.
#
# These are used for static cert distribution (e.g. static clustering) at
# cluster creation time. This will be obsoleted once we implement dynamic
# clustering.
#
# The following certificate pairs are created:
#
#  - ca (the cluster's certificate authority)
#  - server
#  - kubelet
#  - kubecfg (for kubectl)
#
# TODO(roberthbailey): Replace easyrsa with a simple Go program to generate
# the certs that we need.
#
# Assumed vars
#   KUBE_TEMP
#   MASTER_NAME
#
# Vars set:
#   CERT_DIR
#   CA_CERT_BASE64
#   MASTER_CERT_BASE64
#   MASTER_KEY_BASE64
#   KUBELET_CERT_BASE64
#   KUBELET_KEY_BASE64
#   KUBECFG_CERT_BASE64
#   KUBECFG_KEY_BASE64
function create-certs {
  local -r primary_cn="${1}"

  # Determine extra certificate names for master
  local octets=($(echo "${SERVICE_CLUSTER_IP_RANGE}" | sed -e 's|/.*||' -e 's/\./ /g'))
  ((octets[3]+=1))
  local -r service_ip=$(echo "${octets[*]}" | sed 's/ /./g')
  local sans=""
  for extra in $@; do
    if [[ -n "${extra}" ]]; then
      sans="${sans}IP:${extra},"
    fi
  done
  sans="${sans}IP:${service_ip},DNS:kubernetes,DNS:kubernetes.default,DNS:kubernetes.default.svc,DNS:kubernetes.default.svc.${DNS_DOMAIN},DNS:${MASTER_NAME}"

  echo "Generating certs for alternate-names: ${sans}"

  PRIMARY_CN="${primary_cn}" SANS="${sans}" generate-certs

  CERT_DIR="${KUBE_TEMP}/easy-rsa-master/easyrsa3"
  # By default, linux wraps base64 output every 76 cols, so we use 'tr -d' to remove whitespaces.
  # Note 'base64 -w0' doesn't work on Mac OS X, which has different flags.
  CA_CERT_BASE64=$(cat "${CERT_DIR}/pki/ca.crt" | base64 | tr -d '\r\n')
  MASTER_CERT_BASE64=$(cat "${CERT_DIR}/pki/issued/${MASTER_NAME}.crt" | base64 | tr -d '\r\n')
  MASTER_KEY_BASE64=$(cat "${CERT_DIR}/pki/private/${MASTER_NAME}.key" | base64 | tr -d '\r\n')
  KUBELET_CERT_BASE64=$(cat "${CERT_DIR}/pki/issued/kubelet.crt" | base64 | tr -d '\r\n')
  KUBELET_KEY_BASE64=$(cat "${CERT_DIR}/pki/private/kubelet.key" | base64 | tr -d '\r\n')
  KUBECFG_CERT_BASE64=$(cat "${CERT_DIR}/pki/issued/kubecfg.crt" | base64 | tr -d '\r\n')
  KUBECFG_KEY_BASE64=$(cat "${CERT_DIR}/pki/private/kubecfg.key" | base64 | tr -d '\r\n')
}

# Runs the easy RSA commands to generate certificate files.
# The generated files are at ${KUBE_TEMP}/easy-rsa-master/easyrsa3
#
# Assumed vars
#   KUBE_TEMP
#   MASTER_NAME
#   PRIMARY_CN: Primary canonical name
#   SANS: Subject alternate names
#
#
function generate-certs {
  local -r cert_create_debug_output=$(mktemp "${KUBE_TEMP}/cert_create_debug_output.XXX")
  # Note: This was heavily cribbed from make-ca-cert.sh
  (set -x
    cd "${KUBE_TEMP}"
    curl -L -O --connect-timeout 20 --retry 6 --retry-delay 2 https://storage.googleapis.com/kubernetes-release/easy-rsa/easy-rsa.tar.gz
    tar xzf easy-rsa.tar.gz
    cd easy-rsa-master/easyrsa3
    ./easyrsa init-pki
    ./easyrsa --batch "--req-cn=${PRIMARY_CN}@$(date +%s)" build-ca nopass
    ./easyrsa --subject-alt-name="${SANS}" build-server-full "${MASTER_NAME}" nopass
    ./easyrsa build-client-full kubelet nopass
    ./easyrsa build-client-full kubecfg nopass) &>${cert_create_debug_output} || {
    # If there was an error in the subshell, just die.
    # TODO(roberthbailey): add better error handling here
    cat "${cert_create_debug_output}" >&2
    echo "=== Failed to generate certificates: Aborting ===" >&2
    exit 2
  }
}

#
# Using provided master env, extracts value from provided key.
#
# Args:
# $1 master env (kube-env of master; result of calling get-master-env)
# $2 env key to use
function get-env-val() {
  local match=`(echo "${1}" | grep ${2}) || echo ""`
  if [[ -z ${match} ]]; then
    echo ""
  fi
  echo ${match} | cut -d : -f 2 | cut -d \' -f 2
}

# Load the master env by calling get-master-env, and extract important values
function parse-master-env() {
  # Get required master env vars
  local master_env=$(get-master-env)
  KUBELET_TOKEN=$(get-env-val "${master_env}" "KUBELET_TOKEN")
  KUBE_PROXY_TOKEN=$(get-env-val "${master_env}" "KUBE_PROXY_TOKEN")
  CA_CERT_BASE64=$(get-env-val "${master_env}" "CA_CERT")
  EXTRA_DOCKER_OPTS=$(get-env-val "${master_env}" "EXTRA_DOCKER_OPTS")
  KUBELET_CERT_BASE64=$(get-env-val "${master_env}" "KUBELET_CERT")
  KUBELET_KEY_BASE64=$(get-env-val "${master_env}" "KUBELET_KEY")
}
