#!/usr/bin/env bash

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

# Common utilities for kube-up/kube-down

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(cd $(dirname "${BASH_SOURCE[0]}")/.. && pwd)

DEFAULT_KUBECONFIG="${HOME:-.}/.kube/config"

source "${KUBE_ROOT}/hack/lib/util.sh"
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
    "${KUBE_ROOT}/node/${tarball}"
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
#   NODE_BINARY_TAR
#   SERVER_BINARY_TAR
#   KUBE_MANIFESTS_TAR
function find-release-tars() {
  SERVER_BINARY_TAR=$(find-tar kubernetes-server-linux-amd64.tar.gz)
  if [[ "${NUM_WINDOWS_NODES}" -gt "0" ]]; then
    NODE_BINARY_TAR=$(find-tar kubernetes-node-windows-amd64.tar.gz)
  fi

  # This tarball is used by GCI, Ubuntu Trusty, and Container Linux.
  KUBE_MANIFESTS_TAR=
  if [[ "${MASTER_OS_DISTRIBUTION:-}" == "trusty" || "${MASTER_OS_DISTRIBUTION:-}" == "gci" || "${MASTER_OS_DISTRIBUTION:-}" == "ubuntu" ]] || \
     [[ "${NODE_OS_DISTRIBUTION:-}" == "trusty" || "${NODE_OS_DISTRIBUTION:-}" == "gci" || "${NODE_OS_DISTRIBUTION:-}" == "ubuntu" || "${NODE_OS_DISTRIBUTION:-}" == "custom" ]] ; then
    KUBE_MANIFESTS_TAR=$(find-tar kubernetes-manifests.tar.gz)
  fi
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
                    "server auth",
                    "client auth"
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

# Check whether required binaries exist, prompting to download
# if missing.
# If KUBERNETES_SKIP_CONFIRM is set to y, we'll automatically download binaries
# without prompting.
function verify-kube-binaries() {
  if ! "${KUBE_ROOT}/cluster/kubectl.sh" version --client >&/dev/null; then
    echo "!!! kubectl appears to be broken or missing"
    download-release-binaries
  fi
}

# Check whether required release artifacts exist, prompting to download
# if missing.
# If KUBERNETES_SKIP_CONFIRM is set to y, we'll automatically download binaries
# without prompting.
function verify-release-tars() {
  if ! $(find-release-tars); then
    download-release-binaries
  fi
}

# Download release artifacts.
function download-release-binaries() {
  get_binaries_script="${KUBE_ROOT}/cluster/get-kube-binaries.sh"
  local resp="y"
  if [[ ! "${KUBERNETES_SKIP_CONFIRM:-n}" =~ ^[yY]$ ]]; then
    echo "Required release artifacts appear to be missing. Do you wish to download them? [Y/n]"
    read resp
  fi
  if [[ "${resp}" =~ ^[nN]$ ]]; then
    echo "You must download release artifacts to continue. You can use "
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
