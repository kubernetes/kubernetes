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

# !!!EXPERIMENTAL !!! Upgrade script for GCE. Expect this to get
# rewritten in Go in relatively short order, but it allows us to start
# testing the concepts.

set -o errexit
set -o nounset
set -o pipefail

if [[ "${KUBERNETES_PROVIDER:-gce}" != "gce" ]]; then
  echo "!!! ${1} only works on GCE" >&2
  exit 1
fi

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
source "${KUBE_ROOT}/cluster/kube-env.sh"
source "${KUBE_ROOT}/cluster/kube-util.sh"

function usage() {
  echo "!!! EXPERIMENTAL !!!"
  echo ""
  echo "${0} [-M|-N|-P] -l | <release or continuous integration version> | [latest_stable|latest_release|latest_ci]"
  echo "  Upgrades master and nodes by default"
  echo "  -M:  Upgrade master only"
  echo "  -N:  Upgrade nodes only"
  echo "  -P:  Node upgrade prerequisites only (create a new instance template)"
  echo "  -l:  Use local(dev) binaries"
  echo ""
  echo "(... Fetching current release versions ...)"
  echo ""

  # NOTE: IF YOU CHANGE THE FOLLOWING LIST, ALSO UPDATE test/e2e/cluster_upgrade.go
  local latest_release
  local latest_stable
  local latest_ci

  latest_stable=$(gsutil cat gs://kubernetes-release/release/stable.txt)
  latest_release=$(gsutil cat gs://kubernetes-release/release/latest.txt)
  latest_ci=$(gsutil cat gs://kubernetes-release/ci/latest.txt)

  echo "To upgrade to:"
  echo "  latest stable:  ${0} ${latest_stable}"
  echo "  latest release: ${0} ${latest_release}"
  echo "  latest ci:      ${0} ${latest_ci}"
}

function upgrade-master() {
  echo "== Upgrading master to '${SERVER_BINARY_TAR_URL}'. Do not interrupt, deleting master instance. =="

  get-kubeconfig-basicauth
  get-kubeconfig-bearertoken

  detect-master

  # Delete the master instance. Note that the master-pd is created
  # with auto-delete=no, so it should not be deleted.
  gcloud compute instances delete \
    --project "${PROJECT}" \
    --quiet \
    --zone "${ZONE}" \
    "${MASTER_NAME}"

  create-master-instance "${MASTER_NAME}-ip"
  wait-for-master
}

function wait-for-master() {
  echo "== Waiting for new master to respond to API requests =="

  local curl_auth_arg
  if [[ -n ${KUBE_BEARER_TOKEN:-} ]]; then
    curl_auth_arg=(-H "Authorization: Bearer ${KUBE_BEARER_TOKEN}")
  elif [[ -n ${KUBE_PASSWORD:-} ]]; then
    curl_auth_arg=(--user "${KUBE_USER}:${KUBE_PASSWORD}")
  else
    echo "can't get auth credentials for the current master"
    exit 1
  fi

  until curl --insecure "${curl_auth_arg[@]}" --max-time 5 \
    --fail --output /dev/null --silent "https://${KUBE_MASTER_IP}/healthz"; do
    printf "."
    sleep 2
  done

  echo "== Done =="
}

# Perform common upgrade setup tasks
#
# Assumed vars
#   KUBE_VERSION
function prepare-upgrade() {
  ensure-temp-dir
  detect-project
  tars_from_version
}


# Reads kube-env metadata from first node in MINION_NAMES.
#
# Assumed vars:
#   MINION_NAMES
#   PROJECT
#   ZONE
function get-node-env() {
  # TODO(zmerlynn): Make this more reliable with retries.
  gcloud compute --project ${PROJECT} ssh --zone ${ZONE} ${MINION_NAMES[0]} --command \
    "curl --fail --silent -H 'Metadata-Flavor: Google' \
      'http://metadata/computeMetadata/v1/instance/attributes/kube-env'" 2>/dev/null
}

# Using provided node env, extracts value from provided key.
#
# Args:
# $1 node env (kube-env of node; result of calling get-node-env)
# $2 env key to use
function get-env-val() {
  echo "${1}" | grep ${2} | cut -d : -f 2 | cut -d \' -f 2
}

# Assumed vars:
#   KUBE_VERSION
#   MINION_SCOPES
#   NODE_INSTANCE_PREFIX
#   PROJECT
#   ZONE
#
# Vars set:
#   KUBELET_TOKEN
#   KUBE_PROXY_TOKEN
#   CA_CERT_BASE64
#   EXTRA_DOCKER_OPTS
#   KUBELET_CERT_BASE64
#   KUBELET_KEY_BASE64
function upgrade-nodes() {
  prepare-node-upgrade
  do-node-upgrade
}

# prepare-node-upgrade creates a new instance template suitable for upgrading
# to KUBE_VERSION and echos a single line with the name of the new template.
#
# Assumed vars:
#   KUBE_VERSION
#   MINION_SCOPES
#   NODE_INSTANCE_PREFIX
#   PROJECT
#   ZONE
#
# Vars set:
#   SANITIZED_VERSION
#   KUBELET_TOKEN
#   KUBE_PROXY_TOKEN
#   CA_CERT_BASE64
#   EXTRA_DOCKER_OPTS
#   KUBELET_CERT_BASE64
#   KUBELET_KEY_BASE64
function prepare-node-upgrade() {
  echo "== Preparing node upgrade (to ${KUBE_VERSION}). ==" >&2
  SANITIZED_VERSION=$(echo ${KUBE_VERSION} | sed 's/[\.\+]/-/g')

  detect-minion-names

  # TODO(zmerlynn): Refactor setting scope flags.
  local scope_flags=
  if [ -n "${MINION_SCOPES}" ]; then
    scope_flags="--scopes ${MINION_SCOPES}"
  else
    scope_flags="--no-scopes"
  fi

  # Get required node env vars from exiting template.
  local node_env=$(get-node-env)
  KUBELET_TOKEN=$(get-env-val "${node_env}" "KUBELET_TOKEN")
  KUBE_PROXY_TOKEN=$(get-env-val "${node_env}" "KUBE_PROXY_TOKEN")
  CA_CERT_BASE64=$(get-env-val "${node_env}" "CA_CERT")
  EXTRA_DOCKER_OPTS=$(get-env-val "${node_env}" "EXTRA_DOCKER_OPTS")
  KUBELET_CERT_BASE64=$(get-env-val "${node_env}" "KUBELET_CERT")
  KUBELET_KEY_BASE64=$(get-env-val "${node_env}" "KUBELET_KEY")

  # TODO(zmerlynn): How do we ensure kube-env is written in a ${version}-
  #                 compatible way?
  write-node-env

  # TODO(zmerlynn): Get configure-vm script from ${version}. (Must plumb this
  #                 through all create-node-instance-template implementations).
  local template_name=$(get-template-name-from-version ${SANITIZED_VERSION})
  create-node-instance-template "${template_name}"
  # The following is echo'd so that callers can get the template name.
  echo $template_name
  echo "== Finished preparing node upgrade (to ${KUBE_VERSION}). ==" >&2
}

# Prereqs:
# - prepare-node-upgrade should have been called successfully
function do-node-upgrade() {
  echo "== Upgrading nodes to ${KUBE_VERSION}. ==" >&2
  # Do the actual upgrade.
  # NOTE(zmerlynn): If you are changing this gcloud command, update
  #                 test/e2e/cluster_upgrade.go to match this EXACTLY.
  # TODO(zmerlynn): Remove this hack on July 29, 2015, when the migration to
  #                 `gcloud alpha compute rolling-updates` is complete.
  local subgroup="preview"
  local exists=$(gcloud ${subgroup} rolling-updates -h &>/dev/null; echo $?) || true
  if [[ "${exists}" != "0" ]]; then
    subgroup="alpha compute"
  fi
  local template_name=$(get-template-name-from-version ${SANITIZED_VERSION})
  gcloud ${subgroup} rolling-updates \
      --project="${PROJECT}" \
      --zone="${ZONE}" \
      start \
      --group="${NODE_INSTANCE_PREFIX}-group" \
      --template="${template_name}" \
      --instance-startup-timeout=300s \
      --max-num-concurrent-instances=1 \
      --max-num-failed-instances=0 \
      --min-instance-update-time=0s

  # TODO(zmerlynn): Wait for the rolling-update to finish.

  echo "== Finished upgrading nodes to ${KUBE_VERSION}. ==" >&2
}

master_upgrade=true
node_upgrade=true
node_prereqs=false
local_binaries=false

while getopts ":MNPlh" opt; do
  case ${opt} in
    M)
      node_upgrade=false
      ;;
    N)
      master_upgrade=false
      ;;
    P)
      node_prereqs=true
      ;;
    l)
      local_binaries=true
      ;;
    h)
      usage
      exit 0
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      usage
      exit 1
      ;;
  esac
done
shift $((OPTIND-1))

if [[ $# -lt 1 ]] && [[ "${local_binaries}" == "false" ]]; then
  usage
  exit 1
fi

if [[ "${master_upgrade}" == "false" ]] && [[ "${node_upgrade}" == "false" ]]; then
  echo "Can't specify both -M and -N" >&2
  exit 1
fi

if [[ "${local_binaries}" == "false" ]]; then
  set_binary_version ${1}
fi

prepare-upgrade

if [[ "${node_prereqs}" == "true" ]]; then
  prepare-node-upgrade
  exit 0
fi

if [[ "${master_upgrade}" == "true" ]]; then
  upgrade-master
fi

if [[ "${node_upgrade}" == "true" ]]; then
  if [[ "${local_binaries}" == "true" ]]; then
    echo "Upgrading nodes to local binaries is not yet supported." >&2
    exit 1
  else
    upgrade-nodes
  fi
fi

echo "== Validating cluster post-upgrade =="
"${KUBE_ROOT}/cluster/validate-cluster.sh"
