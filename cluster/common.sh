#!/bin/bash

# Copyright 2015 Google Inc. All rights reserved.
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
#
#   KUBE_CERT
#   KUBE_KEY
#   CA_CERT
#   CONTEXT
function create-kubeconfig() {
  local kubectl="${KUBE_ROOT}/cluster/kubectl.sh"

  KUBECONFIG=${KUBECONFIG:-$DEFAULT_KUBECONFIG}
  # KUBECONFIG determines the file we write to, but it may not exist yet
  if [[ ! -e "${KUBECONFIG}" ]]; then
    mkdir -p $(dirname "${KUBECONFIG}")
    touch "${KUBECONFIG}"
  fi
  "${kubectl}" config set-cluster "${CONTEXT}" --server="https://${KUBE_MASTER_IP}" \
                                               --certificate-authority="${CA_CERT}" \
                                               --embed-certs=true
  "${kubectl}" config set-credentials "${CONTEXT}" --username="${KUBE_USER}" \
                                                --password="${KUBE_PASSWORD}" \
                                                --client-certificate="${KUBE_CERT}" \
                                                --client-key="${KUBE_KEY}" \
                                                --embed-certs=true
  "${kubectl}" config set-context "${CONTEXT}" --cluster="${CONTEXT}" --user="${CONTEXT}"
  "${kubectl}" config use-context "${CONTEXT}"  --cluster="${CONTEXT}"

   echo "Wrote config for ${CONTEXT} to ${KUBECONFIG}"
}

# Clear kubeconfig data for a context
# Assumed vars:
#   KUBECONFIG
#   CONTEXT
function clear-kubeconfig() {
  KUBECONFIG=${KUBECONFIG:-$DEFAULT_KUBECONFIG}
  local kubectl="${KUBE_ROOT}/cluster/kubectl.sh"
  "${kubectl}" config unset "clusters.${CONTEXT}"
  "${kubectl}" config unset "users.${CONTEXT}"
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
  KUBECONFIG=${KUBECONFIG:-$DEFAULT_KUBECONFIG}
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
