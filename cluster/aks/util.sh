#!/usr/bin/env bash

# Copyright 2020 The Kubernetes Authors.
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

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/../..

source "${KUBE_ROOT}/cluster/common.sh"
source "${KUBE_ROOT}/hack/lib/util.sh"


function detect-project {
	echo "AKS Provider: detect-project - importing kubeconfig"

	az login --service-principal -u "${AZURE_CLIENT_ID}" -p "${AZURE_CLIENT_SECRET}" -t "${AZURE_TENANT_ID}"
}

# Must ensure that the following ENV vars are set
function detect-master {
	echo "AKS Provider: detect-master" 1>&2
	echo "KUBE_MASTER_IP: ${KUBE_MASTER_IP:-}" 1>&2
	echo "KUBE_MASTER: ${KUBE_MASTER:-}" 1>&2
}
