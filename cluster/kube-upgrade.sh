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

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/cluster/kube-env.sh"
source "${KUBE_ROOT}/cluster/${KUBERNETES_PROVIDER}/util.sh"

function usage() {
  echo "${0} [-m|-n <node id>] <version>"
  echo "  Upgrades master(-m) or specified node(-n)."
  echo "  If the version is not specified will try to use local binaries."
}

upgrade_master=false
upgrade_node=false

while getopts "mn:h" opt; do
  case ${opt} in
    m)
      upgrade_master=true;;
    n)
      upgrade_node=true
      node_to_upgrade="$OPTARG";;
    h)
      usage
      exit 0;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      usage
      exit 1;;
  esac
done
shift $((OPTIND-1))

if [[ "${upgrade_master}" == "false" ]] && [[ "${upgrade_node}" == "false" ]]; then
  echo "No component to upgrade specified"
  usage
  exit 1
fi

verify-prereqs

echo "Upgrading cluster using provider: $KUBERNETES_PROVIDER"

KUBE_VERSION=${1-}
prepare-upgrade $upgrade_node

if [[ "${upgrade_master}" == "true" ]]; then
  echo "Upgrading master to version ${KUBE_VERSION:-"dev"}"
  upgrade-master
fi

if [[ "${upgrade_node}" == "true" ]]; then
  echo "Upgrading node $node_to_upgrade to version ${KUBE_VERSION:-"dev"}"
  upgrade-node $node_to_upgrade
fi

echo "Validating cluster post-upgrade..."
"${KUBE_ROOT}/cluster/validate-cluster.sh"

echo "Done"
