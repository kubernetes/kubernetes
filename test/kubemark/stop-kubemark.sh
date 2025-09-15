#!/usr/bin/env bash

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

# Script that destroys Kubemark cluster and deletes all master resources.

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/../..

source "${KUBE_ROOT}/test/kubemark/skeleton/util.sh"
source "${KUBE_ROOT}/test/kubemark/cloud-provider-config.sh"
source "${KUBE_ROOT}/test/kubemark/${CLOUD_PROVIDER}/util.sh"
source "${KUBE_ROOT}/cluster/kubemark/${CLOUD_PROVIDER}/config-default.sh"

if [[ -f "${KUBE_ROOT}/test/kubemark/${CLOUD_PROVIDER}/shutdown.sh" ]] ; then
  source "${KUBE_ROOT}/test/kubemark/${CLOUD_PROVIDER}/shutdown.sh"
fi

source "${KUBE_ROOT}/cluster/kubemark/util.sh"

KUBECTL="${KUBE_ROOT}/cluster/kubectl.sh"
KUBEMARK_DIRECTORY="${KUBE_ROOT}/test/kubemark"
RESOURCE_DIRECTORY="${KUBEMARK_DIRECTORY}/resources"

detect-project &> /dev/null

"${KUBECTL}" delete --timeout=5m -f "${RESOURCE_DIRECTORY}/addons" --namespace="kubemark" &> /dev/null || true
"${KUBECTL}" delete --timeout=5m -f "${RESOURCE_DIRECTORY}/hollow-node.yaml" --namespace="kubemark" &> /dev/null || true
"${KUBECTL}" delete --timeout=5m -f "${RESOURCE_DIRECTORY}/kubemark-ns.json" &> /dev/null || true

rm -rf "${RESOURCE_DIRECTORY}/addons" \
	"${RESOURCE_DIRECTORY}/kubeconfig.kubemark" \
	"${RESOURCE_DIRECTORY}/hollow-node.yaml"  &> /dev/null || true

delete-kubemark-master
