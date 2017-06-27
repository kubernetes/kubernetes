#!/bin/bash

# Copyright 2017 The Kubernetes Authors.
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

KUBE_ROOT=$(cd $(dirname "${BASH_SOURCE}")/.. && pwd)

DEFAULT_KUBECONFIG="${HOME}/.kube/config"

source "${KUBE_ROOT}/hack/lib/util.sh"
source "${KUBE_ROOT}/cluster/lib/logging.sh"
# KUBE_RELEASE_VERSION_REGEX matches things like "v1.2.3" or "v1.2.3-alpha.4"
#
# NOTE This must match the version_regex in build/common.sh
# kube::release::parse_and_validate_release_version()
KUBE_RELEASE_VERSION_REGEX="^v(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)(-([a-zA-Z0-9]+)\\.(0|[1-9][0-9]*))?$"
KUBE_RELEASE_VERSION_DASHED_REGEX="v(0|[1-9][0-9]*)-(0|[1-9][0-9]*)-(0|[1-9][0-9]*)(-([a-zA-Z0-9]+)-(0|[1-9][0-9]*))?"

# KUBE_CI_VERSION_REGEX matches things like "v1.2.3-alpha.4.56+abcdefg" This
#
# NOTE This must match the version_regex in build/common.sh
# kube::release::parse_and_validate_ci_version()
KUBE_CI_VERSION_REGEX="^v(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)-([a-zA-Z0-9]+)\\.(0|[1-9][0-9]*)(\\.(0|[1-9][0-9]*)\\+[-0-9a-z]*)?$"
KUBE_CI_VERSION_DASHED_REGEX="^v(0|[1-9][0-9]*)-(0|[1-9][0-9]*)-(0|[1-9][0-9]*)-([a-zA-Z0-9]+)-(0|[1-9][0-9]*)(-(0|[1-9][0-9]*)\\+[-0-9a-z]*)?"

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
  OLD_IFS=$IFS
  IFS=':'
  for cfg in ${KUBECONFIG} ; do
    if [[ ! -e "${cfg}" ]]; then
      mkdir -p "$(dirname "${cfg}")"
      touch "${cfg}"
    fi
  done
  IFS=$OLD_IFS

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
  # Unset the current-context before we delete it, as otherwise kubectl errors.
  local cc=$("${kubectl}" config view -o jsonpath='{.current-context}')
  if [[ "${cc}" == "${CONTEXT}" ]]; then
    "${kubectl}" config unset current-context
  fi
  "${kubectl}" config unset "clusters.${CONTEXT}"
  "${kubectl}" config unset "users.${CONTEXT}"
  "${kubectl}" config unset "users.${CONTEXT}-basic-auth"
  "${kubectl}" config unset "contexts.${CONTEXT}"

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
  "${kubectl}" delete deployments --all || true
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
  local sha1sum=""
  if which sha1sum >/dev/null 2>&1; then
    sha1sum="sha1sum"
  else
    sha1sum="shasum -a1"
  fi

  if [[ -z "${KUBE_VERSION-}" ]]; then
    find-release-tars
    upload-server-tars
  elif [[ ${KUBE_VERSION} =~ ${KUBE_RELEASE_VERSION_REGEX} ]]; then
    SERVER_BINARY_TAR_URL="https://storage.googleapis.com/kubernetes-release/release/${KUBE_VERSION}/kubernetes-server-linux-amd64.tar.gz"
    SALT_TAR_URL="https://storage.googleapis.com/kubernetes-release/release/${KUBE_VERSION}/kubernetes-salt.tar.gz"
    # TODO: Clean this up.
    KUBE_MANIFESTS_TAR_URL="${SERVER_BINARY_TAR_URL/server-linux-amd64/manifests}"
    KUBE_MANIFESTS_TAR_HASH=$(curl ${KUBE_MANIFESTS_TAR_URL} --silent --show-error | ${sha1sum} | awk '{print $1}')
  elif [[ ${KUBE_VERSION} =~ ${KUBE_CI_VERSION_REGEX} ]]; then
    SERVER_BINARY_TAR_URL="https://storage.googleapis.com/kubernetes-release-dev/ci/${KUBE_VERSION}/kubernetes-server-linux-amd64.tar.gz"
    SALT_TAR_URL="https://storage.googleapis.com/kubernetes-release-dev/ci/${KUBE_VERSION}/kubernetes-salt.tar.gz"
    # TODO: Clean this up.
    KUBE_MANIFESTS_TAR_URL="${SERVER_BINARY_TAR_URL/server-linux-amd64/manifests}"
    KUBE_MANIFESTS_TAR_HASH=$(curl ${KUBE_MANIFESTS_TAR_URL} --silent --show-error | ${sha1sum} | awk '{print $1}')
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

# Search for the specified tarball in the various known output locations,
# echoing the location if found.
#
# Assumed vars:
#   KUBE_ROOT
#
# Args:
#   $1 name of tarball to search for
function find-tar() {
  local -r tarball=$1
  locations=(
    "${KUBE_ROOT}/server/${tarball}"
    "${KUBE_ROOT}/_output/release-tars/${tarball}"
    "${KUBE_ROOT}/bazel-bin/build/release-tars/${tarball}"
  )
  location=$( (ls -t "${locations[@]}" 2>/dev/null || true) | head -1 )

  if [[ ! -f "${location}" ]]; then
    echo "!!! Cannot find ${tarball}" >&2
    exit 1
  fi
  echo "${location}"
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
  SERVER_BINARY_TAR=$(find-tar kubernetes-server-linux-amd64.tar.gz)
  SALT_TAR=$(find-tar kubernetes-salt.tar.gz)

  # This tarball is used by GCI, Ubuntu Trusty, and Container Linux.
  KUBE_MANIFESTS_TAR=
  if [[ "${MASTER_OS_DISTRIBUTION:-}" == "trusty" || "${MASTER_OS_DISTRIBUTION:-}" == "gci" || "${MASTER_OS_DISTRIBUTION:-}" == "container-linux" || "${MASTER_OS_DISTRIBUTION:-}" == "ubuntu" ]] || \
     [[ "${NODE_OS_DISTRIBUTION:-}" == "trusty" || "${NODE_OS_DISTRIBUTION:-}" == "gci" || "${NODE_OS_DISTRIBUTION:-}" == "container-linux" || "${NODE_OS_DISTRIBUTION:-}" == "ubuntu" ]] ; then
    KUBE_MANIFESTS_TAR=$(find-tar kubernetes-manifests.tar.gz)
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
      "${docker_cmd[@]}" rmi "${KUBE_DOCKER_REGISTRY}/${binary}:${KUBE_IMAGE_TAG}" 2>/dev/null || true
      "${docker_cmd[@]}" tag "gcr.io/google_containers/${binary}:${docker_tag}" "${KUBE_DOCKER_REGISTRY}/${binary}:${KUBE_IMAGE_TAG}"
      "${docker_push_cmd[@]}" push "${KUBE_DOCKER_REGISTRY}/${binary}:${KUBE_IMAGE_TAG}"
    ) &> "${temp_dir}/${binary}-push.log" &
  done

  kube::util::wait-for-jobs || {
    kube::log::error "unable to push images. See ${temp_dir}/*.log for more info."
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
  if [[ "${REGISTER_MASTER_KUBELET:-}" == "true" && -z "${KUBELET_APISERVER:-}" ]]; then
    KUBELET_APISERVER="${MASTER_NAME}"
  fi
  if [[ -z "${KUBERNETES_MASTER_NAME:-}" ]]; then
    KUBERNETES_MASTER_NAME="${MASTER_NAME}"
  fi

  build-kube-env true "${KUBE_TEMP}/master-kube-env.yaml"
  build-kube-master-certs "${KUBE_TEMP}/kube-master-certs.yaml"
}

function write-node-env {
  if [[ -z "${KUBERNETES_MASTER_NAME:-}" ]]; then
    KUBERNETES_MASTER_NAME="${MASTER_NAME}"
  fi

  build-kube-env false "${KUBE_TEMP}/node-kube-env.yaml"
}

function build-kube-master-certs {
  local file=$1
  rm -f ${file}
  cat >$file <<EOF
KUBEAPISERVER_CERT: $(yaml-quote ${KUBEAPISERVER_CERT_BASE64:-})
KUBEAPISERVER_KEY: $(yaml-quote ${KUBEAPISERVER_KEY_BASE64:-})
CA_KEY: $(yaml-quote ${CA_KEY_BASE64:-})
AGGREGATOR_CA_KEY: $(yaml-quote ${AGGREGATOR_CA_KEY_BASE64:-})
REQUESTHEADER_CA_CERT: $(yaml-quote ${REQUESTHEADER_CA_CERT_BASE64:-})
PROXY_CLIENT_CERT: $(yaml-quote ${PROXY_CLIENT_CERT_BASE64:-})
PROXY_CLIENT_KEY: $(yaml-quote ${PROXY_CLIENT_KEY_BASE64:-})
EOF
}

# $1: if 'true', we're building a master yaml, else a node
function build-kube-env {
  local master=$1
  local file=$2

  local server_binary_tar_url=$SERVER_BINARY_TAR_URL
  local salt_tar_url=$SALT_TAR_URL
  local kube_manifests_tar_url="${KUBE_MANIFESTS_TAR_URL:-}"
  if [[ "${master}" == "true" && "${MASTER_OS_DISTRIBUTION}" == "container-linux" ]] || \
     [[ "${master}" == "false" && "${NODE_OS_DISTRIBUTION}" == "container-linux" ]] || \
     [[ "${master}" == "true" && "${MASTER_OS_DISTRIBUTION}" == "ubuntu" ]] || \
     [[ "${master}" == "false" && "${NODE_OS_DISTRIBUTION}" == "ubuntu" ]] ; then
    # TODO: Support fallback .tar.gz settings on Container Linux
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
KUBERNETES_MASTER_NAME: $(yaml-quote ${KUBERNETES_MASTER_NAME})
ALLOCATE_NODE_CIDRS: $(yaml-quote ${ALLOCATE_NODE_CIDRS:-false})
ENABLE_CLUSTER_MONITORING: $(yaml-quote ${ENABLE_CLUSTER_MONITORING:-none})
DOCKER_REGISTRY_MIRROR_URL: $(yaml-quote ${DOCKER_REGISTRY_MIRROR_URL:-})
ENABLE_L7_LOADBALANCING: $(yaml-quote ${ENABLE_L7_LOADBALANCING:-none})
ENABLE_CLUSTER_LOGGING: $(yaml-quote ${ENABLE_CLUSTER_LOGGING:-false})
ENABLE_CLUSTER_UI: $(yaml-quote ${ENABLE_CLUSTER_UI:-false})
ENABLE_NODE_PROBLEM_DETECTOR: $(yaml-quote ${ENABLE_NODE_PROBLEM_DETECTOR:-none})
NODE_PROBLEM_DETECTOR_VERSION: $(yaml-quote ${NODE_PROBLEM_DETECTOR_VERSION:-})
NODE_PROBLEM_DETECTOR_TAR_HASH: $(yaml-quote ${NODE_PROBLEM_DETECTOR_TAR_HASH:-})
ENABLE_NODE_LOGGING: $(yaml-quote ${ENABLE_NODE_LOGGING:-false})
ENABLE_RESCHEDULER: $(yaml-quote ${ENABLE_RESCHEDULER:-false})
LOGGING_DESTINATION: $(yaml-quote ${LOGGING_DESTINATION:-})
ELASTICSEARCH_LOGGING_REPLICAS: $(yaml-quote ${ELASTICSEARCH_LOGGING_REPLICAS:-})
ENABLE_CLUSTER_DNS: $(yaml-quote ${ENABLE_CLUSTER_DNS:-false})
ENABLE_CLUSTER_REGISTRY: $(yaml-quote ${ENABLE_CLUSTER_REGISTRY:-false})
CLUSTER_REGISTRY_DISK: $(yaml-quote ${CLUSTER_REGISTRY_DISK:-})
CLUSTER_REGISTRY_DISK_SIZE: $(yaml-quote ${CLUSTER_REGISTRY_DISK_SIZE:-})
DNS_SERVER_IP: $(yaml-quote ${DNS_SERVER_IP:-})
DNS_DOMAIN: $(yaml-quote ${DNS_DOMAIN:-})
ENABLE_DNS_HORIZONTAL_AUTOSCALER: $(yaml-quote ${ENABLE_DNS_HORIZONTAL_AUTOSCALER:-false})
KUBELET_TOKEN: $(yaml-quote ${KUBELET_TOKEN:-})
KUBE_PROXY_TOKEN: $(yaml-quote ${KUBE_PROXY_TOKEN:-})
NODE_PROBLEM_DETECTOR_TOKEN: $(yaml-quote ${NODE_PROBLEM_DETECTOR_TOKEN:-})
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
SOFTLOCKUP_PANIC: $(yaml-quote ${SOFTLOCKUP_PANIC:-})
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
ENABLE_DEFAULT_STORAGE_CLASS: $(yaml-quote ${ENABLE_DEFAULT_STORAGE_CLASS:-})
ENABLE_APISERVER_BASIC_AUDIT: $(yaml-quote ${ENABLE_APISERVER_BASIC_AUDIT:-})
ENABLE_APISERVER_ADVANCED_AUDIT: $(yaml-quote ${ENABLE_APISERVER_ADVANCED_AUDIT:-})
ENABLE_CACHE_MUTATION_DETECTOR: $(yaml-quote ${ENABLE_CACHE_MUTATION_DETECTOR:-false})
ADVANCED_AUDIT_BACKEND: $(yaml-quote ${ADVANCED_AUDIT_BACKEND:-log})
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
  if [[ "${master}" == "true" && ("${MASTER_OS_DISTRIBUTION}" == "trusty" || "${MASTER_OS_DISTRIBUTION}" == "gci" || "${MASTER_OS_DISTRIBUTION}" == "container-linux") || "${MASTER_OS_DISTRIBUTION}" == "ubuntu" ]] || \
     [[ "${master}" == "false" && ("${NODE_OS_DISTRIBUTION}" == "trusty" || "${NODE_OS_DISTRIBUTION}" == "gci" || "${NODE_OS_DISTRIBUTION}" == "container-linux") || "${NODE_OS_DISTRIBUTION}" = "ubuntu" ]] ; then
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
  if [ -n "${NODE_KUBELET_TEST_ARGS:-}" ]; then
      cat >>$file <<EOF
NODE_KUBELET_TEST_ARGS: $(yaml-quote ${NODE_KUBELET_TEST_ARGS})
EOF
  fi
  if [ -n "${MASTER_KUBELET_TEST_ARGS:-}" ]; then
      cat >>$file <<EOF
MASTER_KUBELET_TEST_ARGS: $(yaml-quote ${MASTER_KUBELET_TEST_ARGS})
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
  if [ -n "${DOCKER_LOG_DRIVER:-}" ]; then
      cat >>$file <<EOF
DOCKER_LOG_DRIVER: $(yaml-quote ${DOCKER_LOG_DRIVER})
EOF
  fi
  if [ -n "${DOCKER_LOG_MAX_SIZE:-}" ]; then
      cat >>$file <<EOF
DOCKER_LOG_MAX_SIZE: $(yaml-quote ${DOCKER_LOG_MAX_SIZE})
EOF
  fi
  if [ -n "${DOCKER_LOG_MAX_FILE:-}" ]; then
      cat >>$file <<EOF
DOCKER_LOG_MAX_FILE: $(yaml-quote ${DOCKER_LOG_MAX_FILE})
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

  if [ -n "${PROVIDER_VARS:-}" ]; then
    local var_name
    local var_value

    for var_name in ${PROVIDER_VARS}; do
      eval "local var_value=\$(yaml-quote \${${var_name}})"
      echo "${var_name}: ${var_value}" >>$file
    done
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
STORAGE_BACKEND: $(yaml-quote ${STORAGE_BACKEND:-etcd3})
STORAGE_MEDIA_TYPE: $(yaml-quote ${STORAGE_MEDIA_TYPE:-})
ENABLE_GARBAGE_COLLECTOR: $(yaml-quote ${ENABLE_GARBAGE_COLLECTOR:-})
ENABLE_LEGACY_ABAC: $(yaml-quote ${ENABLE_LEGACY_ABAC:-})
MASTER_ADVERTISE_ADDRESS: $(yaml-quote ${MASTER_ADVERTISE_ADDRESS:-})
ETCD_CA_KEY: $(yaml-quote ${ETCD_CA_KEY_BASE64:-})
ETCD_CA_CERT: $(yaml-quote ${ETCD_CA_CERT_BASE64:-})
ETCD_PEER_KEY: $(yaml-quote ${ETCD_PEER_KEY_BASE64:-})
ETCD_PEER_CERT: $(yaml-quote ${ETCD_PEER_CERT_BASE64:-})
EOF
    # ETCD_IMAGE (if set) allows to use a custom etcd image.
    if [ -n "${ETCD_IMAGE:-}" ]; then
      cat >>$file <<EOF
ETCD_IMAGE: $(yaml-quote ${ETCD_IMAGE})
EOF
    fi
    # ETCD_VERSION (if set) allows you to use custom version of etcd.
    # The main purpose of using it may be rollback of etcd v3 API,
    # where we need 3.0.* image, but are rolling back to 2.3.7.
    if [ -n "${ETCD_VERSION:-}" ]; then
      cat >>$file <<EOF
ETCD_VERSION: $(yaml-quote ${ETCD_VERSION})
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
    if [ -n "${INITIAL_ETCD_CLUSTER_STATE:-}" ]; then
      cat >>$file <<EOF
INITIAL_ETCD_CLUSTER_STATE: $(yaml-quote ${INITIAL_ETCD_CLUSTER_STATE})
EOF
    fi
    if [ -n "${ETCD_QUORUM_READ:-}" ]; then
      cat >>$file <<EOF
ETCD_QUORUM_READ: $(yaml-quote ${ETCD_QUORUM_READ})
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
  if [[ "${master}" == "true" && "${MASTER_OS_DISTRIBUTION}" == "container-linux" ]] || \
     [[ "${master}" == "false" && "${NODE_OS_DISTRIBUTION}" == "container-linux" ]]; then
    # Container-Linux-only env vars. TODO(yifan): Make them available on other distros.
    cat >>$file <<EOF
KUBERNETES_CONTAINER_RUNTIME: $(yaml-quote ${CONTAINER_RUNTIME:-rkt})
RKT_VERSION: $(yaml-quote ${RKT_VERSION:-})
RKT_PATH: $(yaml-quote ${RKT_PATH:-})
RKT_STAGE1_IMAGE: $(yaml-quote ${RKT_STAGE1_IMAGE:-})
EOF
  fi
  if [[ "${ENABLE_CLUSTER_AUTOSCALER}" == "true" ]]; then
      cat >>$file <<EOF
ENABLE_CLUSTER_AUTOSCALER: $(yaml-quote ${ENABLE_CLUSTER_AUTOSCALER})
AUTOSCALER_MIG_CONFIG: $(yaml-quote ${AUTOSCALER_MIG_CONFIG})
AUTOSCALER_EXPANDER_CONFIG: $(yaml-quote ${AUTOSCALER_EXPANDER_CONFIG})
EOF
  fi

  # Federation specific environment variables.
  if [[ -n "${FEDERATION:-}" ]]; then
    cat >>$file <<EOF
FEDERATION: $(yaml-quote ${FEDERATION})
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
  if which sha1sum >/dev/null 2>&1; then
    sha1sum "$1" | awk '{ print $1 }'
  else
    shasum -a1 "$1" | awk '{ print $1 }'
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

  setup-easyrsa
  PRIMARY_CN="${primary_cn}" SANS="${sans}" generate-certs
  AGGREGATOR_PRIMARY_CN="${primary_cn}" AGGREGATOR_SANS="${sans}" generate-aggregator-certs

  CERT_DIR="${KUBE_TEMP}/easy-rsa-master/easyrsa3"
  # By default, linux wraps base64 output every 76 cols, so we use 'tr -d' to remove whitespaces.
  # Note 'base64 -w0' doesn't work on Mac OS X, which has different flags.
  CA_KEY_BASE64=$(cat "${CERT_DIR}/pki/private/ca.key" | base64 | tr -d '\r\n')
  CA_CERT_BASE64=$(cat "${CERT_DIR}/pki/ca.crt" | base64 | tr -d '\r\n')
  MASTER_CERT_BASE64=$(cat "${CERT_DIR}/pki/issued/${MASTER_NAME}.crt" | base64 | tr -d '\r\n')
  MASTER_KEY_BASE64=$(cat "${CERT_DIR}/pki/private/${MASTER_NAME}.key" | base64 | tr -d '\r\n')
  KUBELET_CERT_BASE64=$(cat "${CERT_DIR}/pki/issued/kubelet.crt" | base64 | tr -d '\r\n')
  KUBELET_KEY_BASE64=$(cat "${CERT_DIR}/pki/private/kubelet.key" | base64 | tr -d '\r\n')
  KUBECFG_CERT_BASE64=$(cat "${CERT_DIR}/pki/issued/kubecfg.crt" | base64 | tr -d '\r\n')
  KUBECFG_KEY_BASE64=$(cat "${CERT_DIR}/pki/private/kubecfg.key" | base64 | tr -d '\r\n')
  KUBEAPISERVER_CERT_BASE64=$(cat "${CERT_DIR}/pki/issued/kube-apiserver.crt" | base64 | tr -d '\r\n')
  KUBEAPISERVER_KEY_BASE64=$(cat "${CERT_DIR}/pki/private/kube-apiserver.key" | base64 | tr -d '\r\n')

  # Setting up an addition directory (beyond pki) as it is the simplest way to
  # ensure we get a different CA pair to sign the proxy-client certs and which
  # we can send CA public key to the user-apiserver to validate communication.
  AGGREGATOR_CERT_DIR="${KUBE_TEMP}/easy-rsa-master/aggregator"
  AGGREGATOR_CA_KEY_BASE64=$(cat "${AGGREGATOR_CERT_DIR}/pki/private/ca.key" | base64 | tr -d '\r\n')
  REQUESTHEADER_CA_CERT_BASE64=$(cat "${AGGREGATOR_CERT_DIR}/pki/ca.crt" | base64 | tr -d '\r\n')
  PROXY_CLIENT_CERT_BASE64=$(cat "${AGGREGATOR_CERT_DIR}/pki/issued/proxy-client.crt" | base64 | tr -d '\r\n')
  PROXY_CLIENT_KEY_BASE64=$(cat "${AGGREGATOR_CERT_DIR}/pki/private/proxy-client.key" | base64 | tr -d '\r\n')
}

function setup-easyrsa {
  local -r cert_create_debug_output=$(mktemp "${KUBE_TEMP}/cert_create_debug_output.XXX")
  # Note: This was heavily cribbed from make-ca-cert.sh
  (set -x
    cd "${KUBE_TEMP}"
    curl -L -O --connect-timeout 20 --retry 6 --retry-delay 2 https://storage.googleapis.com/kubernetes-release/easy-rsa/easy-rsa.tar.gz
    tar xzf easy-rsa.tar.gz
    mkdir easy-rsa-master/kubelet
    cp -r easy-rsa-master/easyrsa3/* easy-rsa-master/kubelet
    mkdir easy-rsa-master/aggregator
    cp -r easy-rsa-master/easyrsa3/* easy-rsa-master/aggregator) &>${cert_create_debug_output} || {
    # If there was an error in the subshell, just die.
    # TODO(roberthbailey): add better error handling here
    cat "${cert_create_debug_output}" >&2
    echo "=== Failed to setup easy-rsa: Aborting ===" >&2
    exit 2
  }
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
    cd "${KUBE_TEMP}/easy-rsa-master/easyrsa3"
    ./easyrsa init-pki
    # this puts the cert into pki/ca.crt and the key into pki/private/ca.key
    ./easyrsa --batch "--req-cn=${PRIMARY_CN}@$(date +%s)" build-ca nopass
    ./easyrsa --subject-alt-name="${SANS}" build-server-full "${MASTER_NAME}" nopass
    ./easyrsa build-client-full kube-apiserver nopass

    kube::util::ensure-cfssl "${KUBE_TEMP}/cfssl"

    # make the config for the signer
    echo '{"signing":{"default":{"expiry":"43800h","usages":["signing","key encipherment","client auth"]}}}' > "ca-config.json"
    # create the kubelet client cert with the correct groups
    echo '{"CN":"kubelet","names":[{"O":"system:nodes"}],"hosts":[""],"key":{"algo":"rsa","size":2048}}' | "${CFSSL_BIN}" gencert -ca=pki/ca.crt -ca-key=pki/private/ca.key -config=ca-config.json - | "${CFSSLJSON_BIN}" -bare kubelet
    mv "kubelet-key.pem" "pki/private/kubelet.key"
    mv "kubelet.pem" "pki/issued/kubelet.crt"
    rm -f "kubelet.csr"

    # Make a superuser client cert with subject "O=system:masters, CN=kubecfg"
    ./easyrsa --dn-mode=org \
      --req-cn=kubecfg --req-org=system:masters \
      --req-c= --req-st= --req-city= --req-email= --req-ou= \
      build-client-full kubecfg nopass) &>${cert_create_debug_output} || {
    # If there was an error in the subshell, just die.
    # TODO(roberthbailey): add better error handling here
    cat "${cert_create_debug_output}" >&2
    echo "=== Failed to generate master certificates: Aborting ===" >&2
    exit 2
  }
}

# Runs the easy RSA commands to generate aggregator certificate files.
# The generated files are at ${KUBE_TEMP}/easy-rsa-master/aggregator
#
# Assumed vars
#   KUBE_TEMP
#   AGGREGATOR_MASTER_NAME
#   AGGREGATOR_PRIMARY_CN: Primary canonical name
#   AGGREGATOR_SANS: Subject alternate names
#
#
function generate-aggregator-certs {
  local -r cert_create_debug_output=$(mktemp "${KUBE_TEMP}/cert_create_debug_output.XXX")
  # Note: This was heavily cribbed from make-ca-cert.sh
  (set -x
    cd "${KUBE_TEMP}/easy-rsa-master/aggregator"
    ./easyrsa init-pki
    # this puts the cert into pki/ca.crt and the key into pki/private/ca.key
    ./easyrsa --batch "--req-cn=${AGGREGATOR_PRIMARY_CN}@$(date +%s)" build-ca nopass
    ./easyrsa --subject-alt-name="${AGGREGATOR_SANS}" build-server-full "${AGGREGATOR_MASTER_NAME}" nopass
    ./easyrsa build-client-full aggregator-apiserver nopass

    kube::util::ensure-cfssl "${KUBE_TEMP}/cfssl"

    # make the config for the signer
    echo '{"signing":{"default":{"expiry":"43800h","usages":["signing","key encipherment","client auth"]}}}' > "ca-config.json"
    # create the aggregator client cert with the correct groups
    echo '{"CN":"aggregator","hosts":[""],"key":{"algo":"rsa","size":2048}}' | "${CFSSL_BIN}" gencert -ca=pki/ca.crt -ca-key=pki/private/ca.key -config=ca-config.json - | "${CFSSLJSON_BIN}" -bare proxy-client
    mv "proxy-client-key.pem" "pki/private/proxy-client.key"
    mv "proxy-client.pem" "pki/issued/proxy-client.crt"
    rm -f "proxy-client.csr"

    # Make a superuser client cert with subject "O=system:masters, CN=kubecfg"
    ./easyrsa --dn-mode=org \
      --req-cn=proxy-clientcfg --req-org=system:aggregator \
      --req-c= --req-st= --req-city= --req-email= --req-ou= \
      build-client-full proxy-clientcfg nopass) &>${cert_create_debug_output} || {
    # If there was an error in the subshell, just die.
    # TODO(roberthbailey): add better error handling here
    cat "${cert_create_debug_output}" >&2
    echo "=== Failed to generate aggregator certificates: Aborting ===" >&2
    exit 2
  }
}

# Run the cfssl command to generates certificate files for etcd service, the
# certificate files will save in $1 directory.
#
# Optional vars:
#   GEN_ETCD_CA_CERT (CA cert encode with base64 and ZIP compression)
#   GEN_ETCD_CA_KEY (CA key encode with base64)
#
# If GEN_ETCD_CA_CERT or GEN_ETCD_CA_KEY is not specified, it will generates certs for CA.
#
# Args:
#   $1 (the directory that certificate files to save)
#   $2 (the ip of etcd member)
#   $3 (the type of etcd certificates, must be one of client, server, peer)
#   $4 (the prefix of the certificate filename, default is $3)
function generate-etcd-cert() {
  local cert_dir=${1}
  local member_ip=${2}
  local type_cert=${3}
  local prefix=${4:-"${type_cert}"}

  local GEN_ETCD_CA_CERT=${GEN_ETCD_CA_CERT:-}
  local GEN_ETCD_CA_KEY=${GEN_ETCD_CA_KEY:-}

  mkdir -p "${cert_dir}"
  pushd "${cert_dir}"

  kube::util::ensure-cfssl .

  if [ ! -r "ca-config.json" ]; then
    cat >ca-config.json <<EOF
{
    "signing": {
        "default": {
            "expiry": "43800h"
        },
        "profiles": {
            "server": {
                "expiry": "43800h",
                "usages": [
                    "signing",
                    "key encipherment",
                    "server auth"
                ]
            },
            "client": {
                "expiry": "43800h",
                "usages": [
                    "signing",
                    "key encipherment",
                    "client auth"
                ]
            },
            "peer": {
                "expiry": "43800h",
                "usages": [
                    "signing",
                    "key encipherment",
                    "server auth",
                    "client auth"
                ]
            }
        }
    }
}
EOF
  fi

  if [ ! -r "ca-csr.json" ]; then
    cat >ca-csr.json <<EOF
{
    "CN": "Kubernetes",
    "key": {
        "algo": "ecdsa",
        "size": 256
    },
    "names": [
        {
            "C": "US",
            "L": "CA",
            "O": "kubernetes.io"
        }
    ]
}
EOF
  fi

  if [[ -n "${GEN_ETCD_CA_CERT}" && -n "${GEN_ETCD_CA_KEY}" ]]; then
    echo "${ca_cert}" | base64 --decode | gunzip > ca.pem
    echo "${ca_key}" | base64 --decode > ca-key.pem
  fi

  if [[ ! -r "ca.pem" || ! -r "ca-key.pem" ]]; then
    ${CFSSL_BIN} gencert -initca ca-csr.json | ${CFSSLJSON_BIN} -bare ca -
  fi

  case "${type_cert}" in
    client)
      echo "Generate client certificates..."
      echo '{"CN":"client","hosts":["*"],"key":{"algo":"ecdsa","size":256}}' \
       | ${CFSSL_BIN} gencert -ca=ca.pem -ca-key=ca-key.pem -config=ca-config.json -profile=client - \
       | ${CFSSLJSON_BIN} -bare "${prefix}"
      ;;
    server)
      echo "Generate server certificates..."
      echo '{"CN":"'${member_ip}'","hosts":[""],"key":{"algo":"ecdsa","size":256}}' \
       | ${CFSSL_BIN} gencert -ca=ca.pem -ca-key=ca-key.pem -config=ca-config.json -profile=server -hostname="${member_ip},127.0.0.1" - \
       | ${CFSSLJSON_BIN} -bare "${prefix}"
      ;;
    peer)
      echo "Generate peer certificates..."
      echo '{"CN":"'${member_ip}'","hosts":[""],"key":{"algo":"ecdsa","size":256}}' \
       | ${CFSSL_BIN} gencert -ca=ca.pem -ca-key=ca-key.pem -config=ca-config.json -profile=peer -hostname="${member_ip},127.0.0.1" - \
       | ${CFSSLJSON_BIN} -bare "${prefix}"
      ;;
    *)
      echo "Unknow, unsupported etcd certs type: ${type_cert}" >&2
      echo "Supported type: client, server, peer" >&2
      exit 2
  esac

  popd
}

#
# Using provided master env, extracts value from provided key.
#
# Args:
# $1 master env (kube-env of master; result of calling get-master-env)
# $2 env key to use
function get-env-val() {
  local match=`(echo "${1}" | grep -E "^${2}:") || echo ""`
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
  NODE_PROBLEM_DETECTOR_TOKEN=$(get-env-val "${master_env}" "NODE_PROBLEM_DETECTOR_TOKEN")
  CA_CERT_BASE64=$(get-env-val "${master_env}" "CA_CERT")
  CA_KEY_BASE64=$(get-env-val "${master_env}" "CA_KEY")
  KUBEAPISERVER_CERT_BASE64=$(get-env-val "${master_env}" "KUBEAPISERVER_CERT")
  KUBEAPISERVER_KEY_BASE64=$(get-env-val "${master_env}" "KUBEAPISERVER_KEY")
  EXTRA_DOCKER_OPTS=$(get-env-val "${master_env}" "EXTRA_DOCKER_OPTS")
  KUBELET_CERT_BASE64=$(get-env-val "${master_env}" "KUBELET_CERT")
  KUBELET_KEY_BASE64=$(get-env-val "${master_env}" "KUBELET_KEY")
  MASTER_CERT_BASE64=$(get-env-val "${master_env}" "MASTER_CERT")
  MASTER_KEY_BASE64=$(get-env-val "${master_env}" "MASTER_KEY")
  AGGREGATOR_CA_KEY_BASE64=$(get-env-val "${master_env}" "AGGREGATOR_CA_KEY")
  REQUESTHEADER_CA_CERT_BASE64=$(get-env-val "${master_env}" "REQUESTHEADER_CA_CERT")
  PROXY_CLIENT_CERT_BASE64=$(get-env-val "${master_env}" "PROXY_CLIENT_CERT")
  PROXY_CLIENT_KEY_BASE64=$(get-env-val "${master_env}" "PROXY_CLIENT_KEY")
}

# Update or verify required gcloud components are installed
# at minimum required version.
# Assumed vars
#   KUBE_PROMPT_FOR_UPDATE
function update-or-verify-gcloud() {
  local sudo_prefix=""
  if [ ! -w $(dirname `which gcloud`) ]; then
    sudo_prefix="sudo"
  fi
  # update and install components as needed
  if [[ "${KUBE_PROMPT_FOR_UPDATE}" == "y" ]]; then
    ${sudo_prefix} gcloud ${gcloud_prompt:-} components install alpha
    ${sudo_prefix} gcloud ${gcloud_prompt:-} components install beta
    ${sudo_prefix} gcloud ${gcloud_prompt:-} components update
  else
    local version=$(gcloud version --format=json)
    python -c'
import json,sys
from distutils import version

minVersion = version.LooseVersion("1.3.0")
required = [ "alpha", "beta", "core" ]
data = json.loads(sys.argv[1])
rel = data.get("Google Cloud SDK")
if rel != "HEAD" and version.LooseVersion(rel) < minVersion:
  print("gcloud version out of date ( < %s )" % minVersion)
  exit(1)
missing = []
for c in required:
  if not data.get(c):
    missing += [c]
if missing:
  for c in missing:
    print ("missing required gcloud component \"{0}\"".format(c))
  exit(1)
    ' """${version}"""
  fi
}

# Check whether required client and server binaries exist, prompting to download
# if missing.
# If KUBERNETES_SKIP_CONFIRM is set to y, we'll automatically download binaries
# without prompting.
function verify-kube-binaries() {
  local missing_binaries=false
  if ! "${KUBE_ROOT}/cluster/kubectl.sh" version --client >&/dev/null; then
    echo "!!! kubectl appears to be broken or missing"
    missing_binaries=true
  fi
  if ! $(find-release-tars); then
    missing_binaries=true
  fi

  if ! "${missing_binaries}"; then
    return
  fi

  get_binaries_script="${KUBE_ROOT}/cluster/get-kube-binaries.sh"
  local resp="y"
  if [[ ! "${KUBERNETES_SKIP_CONFIRM:-n}" =~ ^[yY]$ ]]; then
    echo "Required binaries appear to be missing. Do you wish to download them? [Y/n]"
    read resp
  fi
  if [[ "${resp}" =~ ^[nN]$ ]]; then
    echo "You must download binaries to continue. You can use "
    echo "  ${get_binaries_script}"
    echo "to do this for your automatically."
    exit 1
  fi
  "${get_binaries_script}"
}

# Run pushd without stack output
function pushd() {
  command pushd $@ > /dev/null
}

# Run popd without stack output
function popd() {
  command popd $@ > /dev/null
}
