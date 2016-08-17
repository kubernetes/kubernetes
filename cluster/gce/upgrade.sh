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
source "${KUBE_ROOT}/cluster/kube-util.sh"

function usage() {
  echo "!!! EXPERIMENTAL !!!"
  echo ""
  echo "${0} [-M|-N|-P] -l | <version number or publication>"
  echo "  Upgrades master and nodes by default"
  echo "  -M:  Upgrade master only"
  echo "  -N:  Upgrade nodes only"
  echo "  -P:  Node upgrade prerequisites only (create a new instance template)"
  echo "  -l:  Use local(dev) binaries"
  echo ""
  echo '  Version number or publication is either a proper version number'
  echo '  (e.g. "v1.0.6", "v1.2.0-alpha.1.881+376438b69c7612") or a version'
  echo '  publication of the form <bucket>/<version> (e.g. "release/stable",'
  echo '  "ci/latest-1").  Some common ones are:'
  echo '    - "release/stable"'
  echo '    - "release/latest"'
  echo '    - "ci/latest"'
  echo '  See the docs on getting builds for more information about version publication.'
  echo ""
  echo "(... Fetching current release versions ...)"
  echo ""

  # NOTE: IF YOU CHANGE THE FOLLOWING LIST, ALSO UPDATE test/e2e/cluster_upgrade.go
  local release_stable
  local release_latest
  local ci_latest

  release_stable=$(gsutil cat gs://kubernetes-release/release/stable.txt)
  release_latest=$(gsutil cat gs://kubernetes-release/release/latest.txt)
  ci_latest=$(gsutil cat gs://kubernetes-release-dev/ci/latest.txt)

  echo "Right now, versions are as follows:"
  echo "  release/stable: ${0} ${release_stable}"
  echo "  release/latest: ${0} ${release_latest}"
  echo "  ci/latest:      ${0} ${ci_latest}"
}

function upgrade-master() {
  echo "== Upgrading master to '${SERVER_BINARY_TAR_URL}'. Do not interrupt, deleting master instance. =="

  get-kubeconfig-basicauth
  get-kubeconfig-bearertoken

  detect-master
  parse-master-env

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
  write-cluster-name
  tars_from_version
}

# Reads kube-env metadata from first node in NODE_NAMES.
#
# Assumed vars:
#   NODE_NAMES
#   PROJECT
#   ZONE
function get-node-env() {
  # TODO(zmerlynn): Make this more reliable with retries.
  gcloud compute --project ${PROJECT} ssh --zone ${ZONE} ${NODE_NAMES[0]} --command \
    "curl --fail --silent -H 'Metadata-Flavor: Google' \
      'http://metadata/computeMetadata/v1/instance/attributes/kube-env'" 2>/dev/null
}

# Assumed vars:
#   KUBE_VERSION
#   NODE_SCOPES
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
#   NODE_SCOPES
#   NODE_INSTANCE_PREFIX
#   PROJECT
#   ZONE
#
# Vars set:
#   SANITIZED_VERSION
#   INSTANCE_GROUPS
#   KUBELET_TOKEN
#   KUBE_PROXY_TOKEN
#   CA_CERT_BASE64
#   EXTRA_DOCKER_OPTS
#   KUBELET_CERT_BASE64
#   KUBELET_KEY_BASE64
function prepare-node-upgrade() {
  echo "== Preparing node upgrade (to ${KUBE_VERSION}). ==" >&2
  SANITIZED_VERSION=$(echo ${KUBE_VERSION} | sed 's/[\.\+]/-/g')

  detect-node-names # sets INSTANCE_GROUPS

  # TODO(zmerlynn): Refactor setting scope flags.
  local scope_flags=
  if [ -n "${NODE_SCOPES}" ]; then
    scope_flags="--scopes ${NODE_SCOPES}"
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
  echo "Instance template name: ${template_name}"
  echo "== Finished preparing node upgrade (to ${KUBE_VERSION}). ==" >&2
}

# Prereqs:
# - prepare-node-upgrade should have been called successfully
function do-node-upgrade() {
  echo "== Upgrading nodes to ${KUBE_VERSION}. ==" >&2
  # Do the actual upgrade.
  # NOTE(zmerlynn): If you are changing this gcloud command, update
  #                 test/e2e/cluster_upgrade.go to match this EXACTLY.
  local template_name=$(get-template-name-from-version ${SANITIZED_VERSION})
  local old_templates=()
  local updates=()
  for group in ${INSTANCE_GROUPS[@]}; do
    old_templates+=($(gcloud compute instance-groups managed list \
        --project="${PROJECT}" \
        --zones="${ZONE}" \
        --regexp="${group}" \
        --format='value(instanceTemplate)' || true))
    update=$(gcloud alpha compute rolling-updates \
        --project="${PROJECT}" \
        --zone="${ZONE}" \
        start \
        --group="${group}" \
        --template="${template_name}" \
        --instance-startup-timeout=300s \
        --max-num-concurrent-instances=1 \
        --max-num-failed-instances=0 \
        --min-instance-update-time=0s 2>&1)
    id=$(echo "${update}" | grep "Started" | cut -d '/' -f 11 | cut -d ']' -f 1)
    updates+=("${id}")
  done

  # Wait until rolling updates are finished.
  for update in ${updates[@]}; do
    while true; do
      result=$(gcloud alpha compute rolling-updates \
          --project="${PROJECT}" \
          --zone="${ZONE}" \
          describe \
          ${update} \
          --format='value(status)' || true)
      if [ $result = "ROLLED_OUT" ]; then
        echo "Rolling update ${update} is ${result} state and finished."
        break
      fi
      echo "Rolling update ${update} is still in ${result} state."
      sleep 10
    done
  done

  # Remove the old templates.
  for tmpl in ${old_templates[@]}; do
    gcloud compute instance-templates delete \
        --quiet \
        --project="${PROJECT}" \
        "${tmpl}" || true
  done

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
