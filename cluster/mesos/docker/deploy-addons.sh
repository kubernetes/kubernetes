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

# Deploy the addon services after the cluster is available
# TODO: integrate with or use /cluster/saltbase/salt/kube-addons/kube-addons.sh
# Requires:
#   ENABLE_CLUSTER_DNS (Optional) - 'Y' to deploy kube-dns
#   KUBE_SERVER (Optional) - url to the api server for configuring kube-dns

set -o errexit
set -o nounset
set -o pipefail
set -o errtrace

KUBE_ROOT=$(cd "$(dirname "${BASH_SOURCE}")/../../.." && pwd)
source "${KUBE_ROOT}/cluster/${KUBERNETES_PROVIDER}/${KUBE_CONFIG_FILE-"config-default.sh"}"
kubectl="${KUBE_ROOT}/cluster/kubectl.sh"
bin="$(cd "$(dirname "${BASH_SOURCE}")" && pwd -P)"

# create the kube-system and static-pods namespaces
"${kubectl}" create -f "${KUBE_ROOT}/cluster/mesos/docker/kube-system-ns.yaml"
"${kubectl}" create -f "${KUBE_ROOT}/cluster/mesos/docker/static-pods-ns.yaml"

if [ "${ENABLE_CLUSTER_DNS}" == "true" ]; then
  echo "Deploying DNS Addon" 1>&2
  "${KUBE_ROOT}/third_party/intemp/intemp.sh" -t 'kube-dns' "${bin}/deploy-dns.sh"
fi

if [ "${ENABLE_CLUSTER_UI}" == "true" ]; then
  echo "Deploying UI Addon" 1>&2
  "${bin}/deploy-ui.sh"
fi
