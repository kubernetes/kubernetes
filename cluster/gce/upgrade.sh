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

# VERSION_REGEX matches things like "v0.13.1"
readonly VERSION_REGEX="^v(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)$"

# CI_VERSION_REGEX matches things like "v0.14.1-341-ge0c9d9e"
readonly CI_VERSION_REGEX="^v(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)-(.*)$"

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
  echo "${0} [-M|-N] -l | <release or continuous integration version>"
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

  detect-master
  get-password
  set-master-htpasswd

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

  until curl --insecure --user "${KUBE_USER}:${KUBE_PASSWORD}" --max-time 5 \
    --fail --output /dev/null --silent "https://${KUBE_MASTER_IP}/api/v1beta1/pods"; do
    printf "."
    sleep 2
  done

  echo "== Done =="
}

# Perform common upgrade setup tasks
#
# Assumed vars
#   local_binaries
#   binary_version
function prepare-upgrade() {
  ensure-temp-dir
  detect-project

  if [[ "${local_binaries}" == "true" ]]; then
    find-release-tars
    upload-server-tars
  else
    tars_from_version ${binary_version}
  fi
}

function upgrade-nodes() {
  echo "== Upgrading nodes to ${SERVER_BINARY_TAR_URL}. =="

  detect-minion-names
  get-password
  set-master-htpasswd
  kube-update-nodes upgrade
  echo "== Done =="
}

function tars_from_version() {
  version=${1-}

  if [[ ${version} =~ ${VERSION_REGEX} ]]; then
    SERVER_BINARY_TAR_URL="https://storage.googleapis.com/kubernetes-release/release/${version}/kubernetes-server-linux-amd64.tar.gz"
    SALT_TAR_URL="https://storage.googleapis.com/kubernetes-release/release/${version}/kubernetes-salt.tar.gz"
  elif [[ ${version} =~ ${CI_VERSION_REGEX} ]]; then
    SERVER_BINARY_TAR_URL="https://storage.googleapis.com/kubernetes-release/ci/${version}/kubernetes-server-linux-amd64.tar.gz"
    SALT_TAR_URL="https://storage.googleapis.com/kubernetes-release/ci/${version}/kubernetes-salt.tar.gz"
  else
    echo "!!! Version not provided or version doesn't match regexp" >&2
    exit 1
  fi

  if ! curl -Ss --range 0-1 ${SERVER_BINARY_TAR_URL} >&/dev/null; then
    echo "!!! Can't find release at ${SERVER_BINARY_TAR_URL}" >&2
    exit 1
  fi

  echo "== Release ${version} validated =="
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
  binary_version=${1}
fi

prepare-upgrade

if [[ "${master_upgrade}" == "true" ]]; then
  upgrade-master
fi

if [[ "${node_upgrade}" == "true" ]]; then
  upgrade-nodes
fi

"${KUBE_ROOT}/cluster/validate-cluster.sh"
