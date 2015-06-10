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
source "${KUBE_ROOT}/cluster/${KUBERNETES_PROVIDER}/util.sh"

function usage() {
  echo "!!! EXPERIMENTAL !!!"
  echo ""
  echo "${0} [-M|-N] -l | <release or continuous integration version> | [latest_stable|latest_release|latest_ci]"
  echo "  Upgrades master and nodes by default"
  echo "  -M:  Upgrade master only"
  echo "  -N:  Upgrade nodes only"
  echo "  -l:  Use local(dev) binaries"
  echo ""
  echo "(... Fetching current release versions ...)"
  echo ""

  local latest_release
  local latest_stable
  local latest_ci

  latest_stable=$(gsutil cat gs://kubernetes-release/release/stable.txt)
  latest_release=$(gsutil cat gs://kubernetes-release/release/latest.txt)
  latest_ci=$(gsutil cat gs://kubernetes-release/ci/latest.txt)

  echo "To upgrade to:"
  echo "  latest stable: ${0} ${latest_stable}"
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

# Reads kube-env metadata from master and extracts value from provided key.
#
# Assumed vars:
#   MASTER_NAME
#   ZONE
#
# Args:
# $1 env key to use
function get-env-val() {
  # TODO(mbforbes): Make this more reliable with retries.
  gcloud compute ssh --zone ${ZONE} ${MASTER_NAME} --command \
    "curl --fail --silent -H 'Metadata-Flavor: Google' \
      'http://metadata/computeMetadata/v1/instance/attributes/kube-env'" 2>/dev/null \
    | grep ${1} | cut -d : -f 2 | cut -d \' -f 2
}

# Assumed vars:
#   KUBE_VERSION
#   MINION_SCOPES
#   NODE_INSTANCE_PREFIX
#   PROJECT
#   ZONE
function upgrade-nodes() {
  local sanitized_version=$(echo ${KUBE_VERSION} | sed s/"\."/-/g)
  echo "== Upgrading nodes to ${KUBE_VERSION}. =="

  detect-minion-names

  # TODO(mbforbes): Refactor setting scope flags.
  local -a scope_flags=()
  if (( "${#MINION_SCOPES[@]}" > 0 )); then
    scope_flags=("--scopes" "$(join_csv ${MINION_SCOPES[@]})")
  else
    scope_flags=("--no-scopes")
  fi

  # Get required node tokens.
  KUBELET_TOKEN=$(get-env-val "KUBELET_TOKEN")
  KUBE_PROXY_TOKEN=$(get-env-val "KUBE_PROXY_TOKEN")

  # TODO(mbforbes): How do we ensure kube-env is written in a ${version}-
  #                 compatible way?
  write-node-env
  # TODO(mbforbes): Get configure-vm script from ${version}. (Must plumb this
  #                 through all create-node-instance-template implementations).
  create-node-instance-template ${sanitized_version}

  # Do the actual upgrade.
  gcloud preview rolling-updates start \
      --group "${NODE_INSTANCE_PREFIX}-group" \
      --max-num-concurrent-instances 1 \
      --max-num-failed-instances 0 \
      --project "${PROJECT}" \
      --zone "${ZONE}" \
      --template "${NODE_INSTANCE_PREFIX}-template-${sanitized_version}"

  echo "== Done =="
}

master_upgrade=true
node_upgrade=true
local_binaries=false

while getopts ":MNlh" opt; do
  case ${opt} in
    M)
      node_upgrade=false
      ;;
    N)
      master_upgrade=false
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

if [[ "${master_upgrade}" == "true" ]]; then
  upgrade-master
fi

if [[ "${node_upgrade}" == "true" ]]; then
  if [[ "${local_binaries}" == "true" ]]; then
    echo "Upgrading nodes to local binaries is not yet supported." >&2
  else
    upgrade-nodes
  fi
fi

echo "== Validating cluster post-upgrade =="
"${KUBE_ROOT}/cluster/validate-cluster.sh"
