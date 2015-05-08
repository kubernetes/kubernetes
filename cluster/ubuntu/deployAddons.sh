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

# deploy the add-on services after the cluster is available

set -e

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
source "config-default.sh"

if [ "${ENABLE_CLUSTER_DNS}" == true ]; then
	echo "Deploying DNS on kubernetes"
	sed -e "s/{{ pillar\['dns_replicas'\] }}/${DNS_REPLICAS}/g;s/{{ pillar\['dns_domain'\] }}/${DNS_DOMAIN}/g" ../../cluster/addons/dns/skydns-rc.yaml.in > skydns-rc.yaml
	sed -e "s/{{ pillar\['dns_server'\] }}/${DNS_SERVER_IP}/g" ../../cluster/addons/dns/skydns-svc.yaml.in > skydns-svc.yaml
	# use kubectl to create skydns rc and service
	"${KUBE_ROOT}/cluster/kubectl.sh" create -f skydns-rc.yaml
	"${KUBE_ROOT}/cluster/kubectl.sh" create -f skydns-svc.yaml
	
fi