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

# Script that destroys Kubemark clusters and deletes all GCE resources created for Master
KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..

source "${KUBE_ROOT}/test/kubemark/common.sh"

"${KUBECTL}" delete -f "${RESOURCE_DIRECTORY}/hollow-kubelet.json" &> /dev/null || true
"${KUBECTL}" delete -f "${RESOURCE_DIRECTORY}/addons" &> /dev/null || true
"${KUBECTL}" delete -f "${RESOURCE_DIRECTORY}/kubemark-ns.json" &> /dev/null || true
rm -rf "${RESOURCE_DIRECTORY}/addons"

GCLOUD_COMMON_ARGS="--project ${PROJECT} --zone ${ZONE} --quiet"

gcloud compute instances delete "${MASTER_NAME}" \
    ${GCLOUD_COMMON_ARGS} || true

gcloud compute disks delete "${MASTER_NAME}-pd" \
    ${GCLOUD_COMMON_ARGS} || true

gcloud compute firewall-rules delete "${INSTANCE_PREFIX}-kubemark-master-https" \
	--project "${PROJECT}" \
	--quiet || true

if [ "${SEPARATE_EVENT_MACHINE:-false}" == "true" ]; then
	gcloud compute instances delete "${EVENT_STORE_NAME}" \
    	${GCLOUD_COMMON_ARGS} || true

	gcloud compute disks delete "${EVENT_STORE_NAME}-pd" \
    	${GCLOUD_COMMON_ARGS} || true
fi

rm -rf "${RESOURCE_DIRECTORY}/addons" "${RESOURCE_DIRECTORY}/kubeconfig.loc" &> /dev/null || true
rm "${RESOURCE_DIRECTORY}/ca.crt" "${RESOURCE_DIRECTORY}/kubecfg.crt" "${RESOURCE_DIRECTORY}/kubecfg.key" "${RESOURCE_DIRECTORY}/hollow-node.json" &> /dev/null || true
