#!/bin/bash

# Copyright 2014 The Kubernetes Authors All rights reserved.
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

# Push a new release to the cluster.
#
# This will find the release tar, cause it to be downloaded, unpacked, installed
# and enacted.

set -o errexit
set -o nounset
set -o pipefail

echo "kube-push.sh is currently broken; see https://github.com/kubernetes/kubernetes/issues/17397"
exit 1

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..

if [ -f "${KUBE_ROOT}/cluster/env.sh" ]; then
    source "${KUBE_ROOT}/cluster/env.sh"
fi

source "${KUBE_ROOT}/cluster/kube-util.sh"

function usage() {
  echo "${0} [-m|-n <node id>] <version>"
  echo "  Updates Kubernetes binaries. Can be done for all components (by default), master(-m) or specified node(-n)."
  echo "  If the version is not specified will try to use local binaries."
  echo "  Warning: upgrading single node is experimental"
}

push_to_master=false
push_to_node=false

while getopts "mn:h" opt; do
  case ${opt} in
    m)
      push_to_master=true;;
    n)
      push_to_node=true
      node_id="$OPTARG";;
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

if [[ "${push_to_master}" == "true" ]] && [[ "${push_to_node}" == "true" ]]; then
  echo "Only one of options -m -n should be specified"
  usage
  exit 1
fi

verify-prereqs
KUBE_VERSION=${1-}

if [[ "${push_to_master}" == "false" ]] && [[ "${push_to_node}" == "false" ]]; then
  echo "Updating cluster using provider: $KUBERNETES_PROVIDER"
  kube-push
fi

if [[ "${push_to_master}" == "true" ]]; then
  echo "Updating master to version ${KUBE_VERSION:-"dev"}"
  prepare-push false
  push-master
fi

if [[ "${push_to_node}" == "true" ]]; then
  echo "Updating node $node_id to version ${KUBE_VERSION:-"dev"}"
  prepare-push true
  push-node $node_id
fi

echo "Validating cluster post-push..."

"${KUBE_ROOT}/cluster/validate-cluster.sh"

echo "Done"
