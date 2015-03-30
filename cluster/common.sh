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

# Common utilites for lmktfy-up/lmktfy-down

set -o errexit
set -o nounset
set -o pipefail

LMKTFY_ROOT=$(dirname "${BASH_SOURCE}")/..

# Generate lmktfyconfig data for the created cluster.
# Assumed vars:
#   LMKTFY_USER
#   LMKTFY_PASSWORD
#   LMKTFY_MASTER_IP
#   LMKTFYCONFIG
#
#   LMKTFY_CERT
#   LMKTFY_KEY
#   CA_CERT
#   CONTEXT
function create-lmktfyconfig() {
  local lmktfyctl="${LMKTFY_ROOT}/cluster/lmktfyctl.sh"

  # LMKTFYCONFIG determines the file we write to, but it may not exist yet
  if [[ ! -e "${LMKTFYCONFIG}" ]]; then
    mkdir -p $(dirname "${LMKTFYCONFIG}")
    touch "${LMKTFYCONFIG}"
  fi
  "${lmktfyctl}" config set-cluster "${CONTEXT}" --server="https://${LMKTFY_MASTER_IP}" \
                                               --certificate-authority="${CA_CERT}" \
                                               --embed-certs=true
  "${lmktfyctl}" config set-credentials "${CONTEXT}" --username="${LMKTFY_USER}" \
                                                --password="${LMKTFY_PASSWORD}" \
                                                --client-certificate="${LMKTFY_CERT}" \
                                                --client-key="${LMKTFY_KEY}" \
                                                --embed-certs=true
  "${lmktfyctl}" config set-context "${CONTEXT}" --cluster="${CONTEXT}" --user="${CONTEXT}"
  "${lmktfyctl}" config use-context "${CONTEXT}"  --cluster="${CONTEXT}"

   echo "Wrote config for ${CONTEXT} to ${LMKTFYCONFIG}"
}

# Clear lmktfyconfig data for a context
# Assumed vars:
#   LMKTFYCONFIG
#   CONTEXT
function clear-lmktfyconfig() {
  local lmktfyctl="${LMKTFY_ROOT}/cluster/lmktfyctl.sh"
  "${lmktfyctl}" config unset "clusters.${CONTEXT}"
  "${lmktfyctl}" config unset "users.${CONTEXT}"
  "${lmktfyctl}" config unset "contexts.${CONTEXT}"

  local current
  current=$("${lmktfyctl}" config view -o template --template='{{ index . "current-context" }}')
  if [[ "${current}" == "${CONTEXT}" ]]; then
    "${lmktfyctl}" config unset current-context
  fi

  echo "Cleared config for ${CONTEXT} from ${LMKTFYCONFIG}"
}

# Gets username, password for the current-context in lmktfyconfig, if they exist.
# Assumed vars:
#   LMKTFYCONFIG  # if unset, defaults to global
#
# Vars set:
#   LMKTFY_USER
#   LMKTFY_PASSWORD
#
# LMKTFY_USER,LMKTFY_PASSWORD will be empty if no current-context is set, or
# the current-context user does not exist or contain basicauth entries.
function get-lmktfyconfig-basicauth() {
  # Templates to safely extract the username,password for the current-context
  # user.  The long chain of 'with' commands avoids indexing nil if any of the
  # entries ("current-context", "contexts"."current-context", "users", etc)
  # is missing.
  # Note: we save dot ('.') to $root because the 'with' action overrides it.
  # See http://golang.org/pkg/text/template/.
  local username='{{$root := .}}{{with index $root "current-context"}}{{with index $root "contexts" .}}{{with index . "user"}}{{with index $root "users" .}}{{index . "username"}}{{end}}{{end}}{{end}}{{end}}'
  local password='{{$root := .}}{{with index $root "current-context"}}{{with index $root "contexts" .}}{{with index . "user"}}{{with index $root "users" .}}{{index . "password"}}{{end}}{{end}}{{end}}{{end}}'
  LMKTFY_USER=$("${LMKTFY_ROOT}/cluster/lmktfyctl.sh" config view -o template --template="${username}")
  LMKTFY_PASSWORD=$("${LMKTFY_ROOT}/cluster/lmktfyctl.sh" config view -o template --template="${password}")
  # Handle empty/missing username|password
  if [[ "${LMKTFY_USER}" == '<no value>' || "$LMKTFY_PASSWORD" == '<no value>' ]]; then
    LMKTFY_USER=''
    LMKTFY_PASSWORD=''
  fi
}
