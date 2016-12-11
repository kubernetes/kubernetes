#!/bin/bash

# Copyright 2016 The Kubernetes Authors.
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

# Deploy the Kube-DNS addon

set -o errexit
set -o nounset
set -o pipefail
set -o errtrace

KUBE_ROOT=$(cd "$(dirname "${BASH_SOURCE}")/../../.." && pwd)
source "${KUBE_ROOT}/cluster/${KUBERNETES_PROVIDER}/${KUBE_CONFIG_FILE-"config-default.sh"}"
kubectl="${KUBE_ROOT}/cluster/kubectl.sh"

workspace=$(pwd)

# Process salt pillar templates manually
sed -e "s/{{ pillar\['dns_domain'\] }}/${DNS_DOMAIN}/g" "${KUBE_ROOT}/cluster/addons/dns/kubedns-controller.yaml.in" > "${workspace}/kubedns-controller.yaml"
sed -e "s/{{ pillar\['dns_server'\] }}/${DNS_SERVER_IP}/g" "${KUBE_ROOT}/cluster/addons/dns/kubedns-svc.yaml.in" > "${workspace}/kubedns-svc.yaml"

# Federation specific values.
if [[ "${FEDERATION:-}" == "true" ]]; then
  FEDERATIONS_DOMAIN_MAP="${FEDERATIONS_DOMAIN_MAP:-}"
  if [[ -z "${FEDERATIONS_DOMAIN_MAP}" && -n "${FEDERATION_NAME:-}" && -n "${DNS_ZONE_NAME:-}" ]]; then
    FEDERATIONS_DOMAIN_MAP="${FEDERATION_NAME}=${DNS_ZONE_NAME}"
  fi
  if [[ -n "${FEDERATIONS_DOMAIN_MAP}" ]]; then
    sed -i -e "s/{{ pillar\['federations_domain_map'\] }}/- --federations=${FEDERATIONS_DOMAIN_MAP}/g" "${workspace}/kubedns-controller.yaml"
  else
    sed -i -e "/{{ pillar\['federations_domain_map'\] }}/d" "${workspace}/kubedns-controller.yaml"
  fi
else
  sed -i -e "/{{ pillar\['federations_domain_map'\] }}/d" "${workspace}/kubedns-controller.yaml"
fi

# Use kubectl to create kube-dns controller and service
"${kubectl}" create -f "${workspace}/kubedns-controller.yaml"
"${kubectl}" create -f "${workspace}/kubedns-svc.yaml"
