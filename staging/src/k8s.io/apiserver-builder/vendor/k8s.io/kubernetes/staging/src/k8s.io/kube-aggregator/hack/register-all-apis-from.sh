#!/bin/bash

# Copyright 2017 The Kubernetes Authors.
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

set -o errexit
set -o nounset
set -o pipefail


if LANG=C sed --help 2>&1 | grep -q GNU; then
  SED="sed"
elif which gsed &>/dev/null; then
  SED="gsed"
else
  echo "Failed to find GNU sed as sed or gsed. If you are on Mac: brew install gnu-sed." >&2
  exit 1
fi

dir=$(mktemp -d "${TMPDIR:-/tmp/}$(basename 0).XXXXXXXXXXXX")
# Register function to be called on EXIT to remove generated binary.
function cleanup {
  rm -rf "${dir}"
}
trap cleanup EXIT


scriptDir=$(dirname "${BASH_SOURCE}")

# this uses discovery from a kube-like API server to register ALL the API versions that server provides
# first argument is reference to kube-config file that points the API server you're adding from
# second argument is the service namespace
# third argument is the service name
# fourth argument is reference to kube-config file that points to the aggregator you're using

FROM_KUBECONFIG=${1}
SERVICE_NAMESPACE=${2}
SERVICE_NAME=${3}
AGG_KUBECONFIG=${4}


caBundle=$(base64 /var/run/kubernetes/server-ca.crt | awk 'BEGIN{ORS="";} {print}')

# if we have a /api endpoint, then we need to register that
if kubectl --kubeconfig=${FROM_KUBECONFIG} get --raw / | grep -q /api/v1; then
	group=""
	version="v1"
	resourceName=${version}.${group}
	resourceFileName=${dir}/${resourceName}.yaml
	cp ${scriptDir}/apiservice-template.yaml ${resourceFileName}
	${SED} -i "s/RESOURCE_NAME/${resourceName}/" ${resourceFileName}
	${SED} -i "s/API_GROUP/${group}/" ${resourceFileName}
	${SED} -i "s/API_VERSION/${version}/" ${resourceFileName}
	${SED} -i "s/SERVICE_NAMESPACE/${SERVICE_NAMESPACE}/" ${resourceFileName}
	${SED} -i "s/SERVICE_NAME/${SERVICE_NAME}/" ${resourceFileName}
	${SED} -i "s/CA_BUNDLE/${caBundle}/" ${resourceFileName}
	echo "registering ${resourceName} using ${resourceFileName}"

	kubectl --kubeconfig=${AGG_KUBECONFIG} create -f ${resourceFileName}
fi

groupVersions=( $(kubectl --kubeconfig=${FROM_KUBECONFIG} get --raw / | grep /apis/ | sed 's/",.*//' | sed 's|.*"/apis/||' | grep '/') )

for groupVersion in "${groupVersions[@]}"; do
	group=$(echo $groupVersion | awk -F/ '{print $1}')
	version=$(echo $groupVersion | awk -F/ '{print $2}')
	resourceName=${version}.${group}
	resourceFileName=${dir}/${resourceName}.yaml
	cp ${scriptDir}/apiservice-template.yaml ${resourceFileName}
	${SED} -i "s/RESOURCE_NAME/${resourceName}/" ${resourceFileName}
	${SED} -i "s/API_GROUP/${group}/" ${resourceFileName}
	${SED} -i "s/API_VERSION/${version}/" ${resourceFileName}
	${SED} -i "s/SERVICE_NAMESPACE/${SERVICE_NAMESPACE}/" ${resourceFileName}
	${SED} -i "s/SERVICE_NAME/${SERVICE_NAME}/" ${resourceFileName}
	${SED} -i "s/CA_BUNDLE/${caBundle}/" ${resourceFileName}
	echo "registering ${resourceName} using ${resourceFileName}"

	kubectl --kubeconfig=${AGG_KUBECONFIG} create -f ${resourceFileName}
done
