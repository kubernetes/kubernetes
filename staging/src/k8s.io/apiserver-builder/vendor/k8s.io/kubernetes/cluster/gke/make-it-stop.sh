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

echo "This is NOT a production-ready tool.\n\
IT'S A HACKY, BEST-EFFORT WAY TO \"STOP\" CREATION OF THE GKE CLUSTER."
read -n 1 -p "Are you sure you want to proceed (y/N)?: " decision
echo ""
if [[ "${decision}" != "y" ]]; then
	echo "Aborting..."
	exit 0
fi

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..

if [ -f "${KUBE_ROOT}/cluster/env.sh" ]; then
    source "${KUBE_ROOT}/cluster/env.sh"
fi

source "${KUBE_ROOT}/cluster/gke/util.sh"
STAGING_ENDPOINT="CLOUDSDK_API_ENDPOINT_OVERRIDES_CONTAINER=https://staging-container.sandbox.googleapis.com/"

detect-project
cluster=$(gcloud container operations list "--project=${PROJECT}" | grep "CREATE_CLUSTER" | grep "RUNNING" || true)
if [ -z "${cluster}" ]; then
	echo "Couldn't find any cluster being created in production environment. Trying staging..."
	cluster=$(env ${STAGING_ENDPOINT} gcloud container operations list "--project=${PROJECT}" | grep "CREATE_CLUSTER" | grep "RUNNING" || true)
fi

if [ -z "${cluster}" ]; then
	echo "No cluster creation in progress found. Aborting."
	exit 0
fi

zone=$(echo "${cluster}" | tr -s "[:blank:]" | cut -f3 -d" ")
cluster_name=$(echo "${cluster}" | tr -s "[:blank:]" | cut -f4 -d" ")
gcloud="gcloud"
if [ "${zone}" == "us-east1-a" ]; then
	gcloud="env ${STAGING_ENDPOINT} gcloud"
fi

migs=$(${gcloud} compute instance-groups managed list --project=${PROJECT} --zones=${zone} | grep "gke-${cluster_name}" | cut -f1 -d" ")
echo "Managed instance groups for cluster ${cluster_name}: ${migs}"
for mig in ${migs}; do
	echo "Resizing ${mig}..."
	${gcloud} compute instance-groups managed resize --project="${PROJECT}" --zone="${zone}" "${mig}" --size=1
done

echo "All managed instance groups resized to 1. Cluster creation operation should end soon, and you will be be able to delete the cluster."
