#!/usr/bin/env bash

# Copyright 2018 The Kubernetes Authors.
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

# Script that destroys the clusters used, namespace, and deployment.

KUBECTL=kubectl
KUBEMARK_DIRECTORY="${KUBE_ROOT}/test/kubemark"
RESOURCE_DIRECTORY="${KUBEMARK_DIRECTORY}/resources"

# Login to cloud services
complete-login

# Remove resources created for kubemark
echo -e "${color_yellow}REMOVING RESOURCES${color_norm}"
spawn-config
"${KUBECTL}" delete -f "${RESOURCE_DIRECTORY}/addons" &> /dev/null || true
"${KUBECTL}" delete -f "${RESOURCE_DIRECTORY}/hollow-node.yaml" &> /dev/null || true
"${KUBECTL}" delete -f "${RESOURCE_DIRECTORY}/kubemark-ns.json" &> /dev/null || true
rm -rf "${RESOURCE_DIRECTORY}/addons" 
	"${RESOURCE_DIRECTORY}/hollow-node.yaml" &> /dev/null || true

# Remove clusters, namespaces, and deployments
delete-clusters
if [[ -f "${RESOURCE_DIRECTORY}/iks-namespacelist.sh" ]] ; then
  bash ${RESOURCE_DIRECTORY}/iks-namespacelist.sh
  rm -f ${RESOURCE_DIRECTORY}/iks-namespacelist.sh
fi
echo -e "${color_blue}EXECUTION COMPLETE${color_norm}"
exit 0
