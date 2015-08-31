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
source "${KUBE_ROOT}/cluster/${KUBERNETES_PROVIDER}/common/bin/util-temp-dir.sh"
kubectl="${KUBE_ROOT}/cluster/kubectl.sh"


function deploy_dns {
  echo "Deploying DNS Addon" 1>&2
  local workspace=$(pwd)

  # Process salt pillar templates manually
  sed -e "s/{{ pillar\['dns_replicas'\] }}/${DNS_REPLICAS}/g;s/{{ pillar\['dns_domain'\] }}/${DNS_DOMAIN}/g" "${KUBE_ROOT}/cluster/addons/dns/skydns-rc.yaml.in" > "${workspace}/skydns-rc.yaml"
  sed -e "s/{{ pillar\['dns_server'\] }}/${DNS_SERVER_IP}/g" "${KUBE_ROOT}/cluster/addons/dns/skydns-svc.yaml.in" > "${workspace}/skydns-svc.yaml"

  # Use kubectl to create skydns rc and service
  "${kubectl}" create -f "${workspace}/skydns-rc.yaml"
  "${kubectl}" create -f "${workspace}/skydns-svc.yaml"
}

function deploy_ui {
  echo "Deploying UI Addon" 1>&2

  # Use kubectl to create ui rc and service
  "${kubectl}" create -f "${KUBE_ROOT}/cluster/addons/kube-ui/kube-ui-rc.yaml"
  "${kubectl}" create -f "${KUBE_ROOT}/cluster/addons/kube-ui/kube-ui-svc.yaml"
}

# create the kube-system namespace
"${kubectl}" create -f "${KUBE_ROOT}/cluster/mesos/docker/kube-system-ns.yaml"

if [ "${ENABLE_CLUSTER_DNS}" == true ]; then
  cluster::mesos::docker::run_in_temp_dir 'k8sm-dns' 'deploy_dns'
fi

if [ "${ENABLE_CLUSTER_UI}" == true ]; then
  deploy_ui
fi
