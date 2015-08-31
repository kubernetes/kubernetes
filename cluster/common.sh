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

# Common utilites for kube-up/kube-down

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..

DEFAULT_KUBECONFIG="${HOME}/.kube/config"

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
# The following can be omitted for --insecure-skip-tls-verify
#   KUBE_CERT
#   KUBE_KEY
#   CA_CERT
function create-kubeconfig() {
  local kubectl="${KUBE_ROOT}/cluster/kubectl.sh"

  export KUBECONFIG=${KUBECONFIG:-$DEFAULT_KUBECONFIG}
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

  "${kubectl}" config set-cluster "${CONTEXT}" "${cluster_args[@]}"
  if [[ -n "${user_args[@]:-}" ]]; then
    "${kubectl}" config set-credentials "${CONTEXT}" "${user_args[@]}"
  fi
  "${kubectl}" config set-context "${CONTEXT}" --cluster="${CONTEXT}" --user="${CONTEXT}"
  "${kubectl}" config use-context "${CONTEXT}"  --cluster="${CONTEXT}"

  # If we have a bearer token, also create a credential entry with basic auth
  # so that it is easy to discover the basic auth password for your cluster
  # to use in a web browser.
  if [[ ! -z "${KUBE_BEARER_TOKEN:-}" && ! -z "${KUBE_USER:-}" && ! -z "${KUBE_PASSWORD:-}" ]]; then
    "${kubectl}" config set-credentials "${CONTEXT}-basic-auth" "--username=${KUBE_USER}" "--password=${KUBE_PASSWORD}"
  fi

   echo "Wrote config for ${CONTEXT} to ${KUBECONFIG}"
}

# Clear kubeconfig data for a context
# Assumed vars:
#   KUBECONFIG
#   CONTEXT
function clear-kubeconfig() {
  export KUBECONFIG=${KUBECONFIG:-$DEFAULT_KUBECONFIG}
  local kubectl="${KUBE_ROOT}/cluster/kubectl.sh"
  "${kubectl}" config unset "clusters.${CONTEXT}"
  "${kubectl}" config unset "users.${CONTEXT}"
  "${kubectl}" config unset "users.${CONTEXT}-basic-auth"
  "${kubectl}" config unset "contexts.${CONTEXT}"

  local current
  current=$("${kubectl}" config view -o template --template='{{ index . "current-context" }}')
  if [[ "${current}" == "${CONTEXT}" ]]; then
    "${kubectl}" config unset current-context
  fi

  echo "Cleared config for ${CONTEXT} from ${KUBECONFIG}"
}

# Gets username, password for the current-context in kubeconfig, if they exist.
# Assumed vars:
#   KUBECONFIG  # if unset, defaults to global
#
# Vars set:
#   KUBE_USER
#   KUBE_PASSWORD
#
# KUBE_USER,KUBE_PASSWORD will be empty if no current-context is set, or
# the current-context user does not exist or contain basicauth entries.
function get-kubeconfig-basicauth() {
  export KUBECONFIG=${KUBECONFIG:-$DEFAULT_KUBECONFIG}
  # Templates to safely extract the username,password for the current-context
  # user.  The long chain of 'with' commands avoids indexing nil if any of the
  # entries ("current-context", "contexts"."current-context", "users", etc)
  # is missing.
  # Note: we save dot ('.') to $root because the 'with' action overrides it.
  # See http://golang.org/pkg/text/template/.
  local username='{{$dot := .}}{{with $ctx := index $dot "current-context"}}{{range $element := (index $dot "contexts")}}{{ if eq .name $ctx }}{{ with $user := .context.user }}{{range $element := (index $dot "users")}}{{ if eq .name $user }}{{ index . "user" "username" }}{{end}}{{end}}{{end}}{{end}}{{end}}{{end}}'
  local password='{{$dot := .}}{{with $ctx := index $dot "current-context"}}{{range $element := (index $dot "contexts")}}{{ if eq .name $ctx }}{{ with $user := .context.user }}{{range $element := (index $dot "users")}}{{ if eq .name $user }}{{ index . "user" "password" }}{{end}}{{end}}{{end}}{{end}}{{end}}{{end}}'
  KUBE_USER=$("${KUBE_ROOT}/cluster/kubectl.sh" config view -o template --template="${username}")
  KUBE_PASSWORD=$("${KUBE_ROOT}/cluster/kubectl.sh" config view -o template --template="${password}")
  # Handle empty/missing username|password
  if [[ "${KUBE_USER}" == '<no value>' || "$KUBE_PASSWORD" == '<no value>' ]]; then
    KUBE_USER=''
    KUBE_PASSWORD=''
  fi
}

# Generate basic auth user and password.

# Vars set:
#   KUBE_USER
#   KUBE_PASSWORD
function gen-kube-basicauth() {
    KUBE_USER=admin
    KUBE_PASSWORD=$(python -c 'import string,random; print "".join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(16))')
}

# Get the bearer token for the current-context in kubeconfig if one exists.
# Assumed vars:
#   KUBECONFIG  # if unset, defaults to global
#
# Vars set:
#   KUBE_BEARER_TOKEN
#
# KUBE_BEARER_TOKEN will be empty if no current-context is set, or the
# current-context user does not exist or contain a bearer token entry.
function get-kubeconfig-bearertoken() {
  export KUBECONFIG=${KUBECONFIG:-$DEFAULT_KUBECONFIG}
  # Template to safely extract the token for the current-context user.
  # The long chain of 'with' commands avoids indexing nil if any of the
  # entries ("current-context", "contexts"."current-context", "users", etc)
  # is missing.
  # Note: we save dot ('.') to $root because the 'with' action overrides it.
  # See http://golang.org/pkg/text/template/.
  local token='{{$dot := .}}{{with $ctx := index $dot "current-context"}}{{range $element := (index $dot "contexts")}}{{ if eq .name $ctx }}{{ with $user := .context.user }}{{range $element := (index $dot "users")}}{{ if eq .name $user }}{{ index . "user" "token" }}{{end}}{{end}}{{end}}{{end}}{{end}}{{end}}'
  KUBE_BEARER_TOKEN=$("${KUBE_ROOT}/cluster/kubectl.sh" config view -o template --template="${token}")
  # Handle empty/missing token
  if [[ "${KUBE_BEARER_TOKEN}" == '<no value>' ]]; then
    KUBE_BEARER_TOKEN=''
  fi
}

# Generate bearer token.
#
# Vars set:
#   KUBE_BEARER_TOKEN
function gen-kube-bearertoken() {
    KUBE_BEARER_TOKEN=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
}

# Get the master IP for the current-context in kubeconfig if one exists.
#
# Assumed vars:
#   KUBECONFIG  # if unset, defaults to global
#
# Vars set:
#   KUBE_MASTER_URL
#
# KUBE_MASTER_URL will be empty if no current-context is set, or the
# current-context user does not exist or contain a server entry.
function detect-master-from-kubeconfig() {
  export KUBECONFIG=${KUBECONFIG:-$DEFAULT_KUBECONFIG}
  # Template to safely extract the server for the current-context cluster.
  # The long chain of 'with' commands avoids indexing nil if any of the
  # entries ("current-context", "contexts"."current-context", "users", etc)
  # is missing.
  # Note: we save dot ('.') to $root because the 'with' action overrides it.
  # See http://golang.org/pkg/text/template/.
  local server_tpl='{{$dot := .}}{{with $ctx := index $dot "current-context"}}{{range $element := (index $dot "contexts")}}{{ if eq .name $ctx }}{{ with $cluster := .context.cluster }}{{range $element := (index $dot "clusters")}}{{ if eq .name $cluster }}{{ index . "cluster" "server" }}{{end}}{{end}}{{end}}{{end}}{{end}}{{end}}'
  KUBE_MASTER_URL=$("${KUBE_ROOT}/cluster/kubectl.sh" config view -o template --template="${server_tpl}")
  # Handle empty/missing server
  if [[ "${KUBE_MASTER_URL}" == '<no value>' ]]; then
    KUBE_MASTER_URL=''
  fi
}

# Sets KUBE_VERSION variable to the version passed in as an argument, or if argument is
# latest_stable, latest_release, or latest_ci fetches and sets the corresponding version number
#
# Args:
#   $1 version string from command line
# Vars set:
#   KUBE_VERSION
function set_binary_version() {
  if [[ "${1}" == "latest_stable" ]]; then
    KUBE_VERSION=$(gsutil cat gs://kubernetes-release/release/stable.txt)
    echo "Using latest stable version: ${KUBE_VERSION}" >&2
  elif [[ "${1}" == "latest_release" ]]; then
    KUBE_VERSION=$(gsutil cat gs://kubernetes-release/release/latest.txt)
    echo "Using latest release version: ${KUBE_VERSION}" >&2
  elif [[ "${1}" == "latest_ci" ]]; then
    KUBE_VERSION=$(gsutil cat gs://kubernetes-release/ci/latest.txt)
    echo "Using latest ci version: ${KUBE_VERSION}" >&2
  else
    KUBE_VERSION=${1}
  fi
}

# Figure out which binary use on the server and assure it is available.
# If KUBE_VERSION is specified use binaries specified by it, otherwise
# use local dev binaries.
#
# Assumed vars:
#   PROJECT
# Vars set:
#   SERVER_BINARY_TAR_URL
#   SERVER_BINARY_TAR_HASH
#   SALT_TAR_URL
#   SALT_TAR_HASH
function tars_from_version() {
  if [[ -z "${KUBE_VERSION-}" ]]; then
    find-release-tars
    upload-server-tars
  elif [[ ${KUBE_VERSION} =~ ${KUBE_VERSION_REGEX} ]]; then
    SERVER_BINARY_TAR_URL="https://storage.googleapis.com/kubernetes-release/release/${KUBE_VERSION}/kubernetes-server-linux-amd64.tar.gz"
    SALT_TAR_URL="https://storage.googleapis.com/kubernetes-release/release/${KUBE_VERSION}/kubernetes-salt.tar.gz"
  elif [[ ${KUBE_VERSION} =~ ${KUBE_CI_VERSION_REGEX} ]]; then
    SERVER_BINARY_TAR_URL="https://storage.googleapis.com/kubernetes-release/ci/${KUBE_VERSION}/kubernetes-server-linux-amd64.tar.gz"
    SALT_TAR_URL="https://storage.googleapis.com/kubernetes-release/ci/${KUBE_VERSION}/kubernetes-salt.tar.gz"
  else
    echo "Version doesn't match regexp" >&2
    exit 1
  fi
  until SERVER_BINARY_TAR_HASH=$(curl --fail --silent "${SERVER_BINARY_TAR_URL}.sha1"); do
    echo "Failure trying to curl release .sha1"
  done
  until SALT_TAR_HASH=$(curl --fail --silent "${SALT_TAR_URL}.sha1"); do
    echo "Failure trying to curl Salt tar .sha1"
  done

  if ! curl -Ss --range 0-1 "${SERVER_BINARY_TAR_URL}" >&/dev/null; then
    echo "Can't find release at ${SERVER_BINARY_TAR_URL}" >&2
    exit 1
  fi
  if ! curl -Ss --range 0-1 "${SALT_TAR_URL}" >&/dev/null; then
    echo "Can't find Salt tar at ${SALT_TAR_URL}" >&2
    exit 1
  fi
}
