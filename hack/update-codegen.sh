#!/bin/bash

# Copyright 2014 The Kubernetes Authors.
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

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::golang::setup_env

BUILD_TARGETS=(
  cmd/libs/go2idl/client-gen
  cmd/libs/go2idl/set-gen
)
make -C "${KUBE_ROOT}" WHAT="${BUILD_TARGETS[*]}"

clientgen=$(kube::util::find-binary "client-gen")
setgen=$(kube::util::find-binary "set-gen")

# Please do not add any logic to this shell script. Add logic to the go code
# that generates the set-gen program.
#

GROUP_VERSIONS=(${KUBE_AVAILABLE_GROUP_VERSIONS})
GV_DIRS=()
SEEN_GROUPS=","
for gv in "${GROUP_VERSIONS[@]}"; do
	# add items, but strip off any leading apis/ you find to match command expectations
	api_dir=$(kube::util::group-version-to-pkg-path "${gv}")
	pkg_dir=${api_dir#apis/}

	# don't add a version for a group you've already seen
	group=${pkg_dir%%/*}
	if [[ "${SEEN_GROUPS}" == *",${group}."* ]]; then
		continue
	fi
	SEEN_GROUPS="${SEEN_GROUPS},${group}."

	# skip groups that aren't being served, clients for these don't matter
    if [[ " ${KUBE_NONSERVER_GROUP_VERSIONS} " == *" ${gv} "* ]]; then
      continue
    fi

	GV_DIRS+=("${pkg_dir}")
done
# delimit by commas for the command
GV_DIRS_CSV=$(IFS=',';echo "${GV_DIRS[*]// /,}";IFS=$)

# This can be called with one flag, --verify-only, so it works for both the
# update- and verify- scripts.
${clientgen} "$@"
${clientgen} -t "$@"
${clientgen} --clientset-name="release_1_5" --input="${GV_DIRS_CSV}" "$@"
# Clientgen for federation clientset.
${clientgen} --clientset-name=federation_internalclientset --clientset-path=k8s.io/kubernetes/federation/client/clientset_generated --input="../../federation/apis/federation/","api/","extensions/" --included-types-overrides="api/Service,api/Namespace,extensions/ReplicaSet,api/Secret,extensions/Ingress,extensions/Deployment,extensions/DaemonSet,api/ConfigMap,api/Event"   "$@"
${clientgen} --clientset-name=federation_release_1_5 --clientset-path=k8s.io/kubernetes/federation/client/clientset_generated --input="../../federation/apis/federation/v1beta1","api/v1","extensions/v1beta1" --included-types-overrides="api/v1/Service,api/v1/Namespace,extensions/v1beta1/ReplicaSet,api/v1/Secret,extensions/v1beta1/Ingress,extensions/v1beta1/Deployment,extensions/v1beta1/DaemonSet,api/v1/ConfigMap,api/v1/Event"   "$@"
${setgen} "$@"

# You may add additional calls of code generators like set-gen above.
