#!/usr/bin/env bash

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

# This script will build the hyperkube image and push it to the repository
# referred to by KUBE_DOCKER_REGISTRY and KUBE_DOCKER_OWNER. The image will
# be given a version tag with the value from KUBE_DOCKER_VERSION.
# e.g. run as: 
# KUBE_DOCKER_REGISTRY=localhost:5000 KUBE_DOCKER_OWNER=liyi \
# KUBE_DOCKER_VERSION=1.3.0-dev ./hack/dev-push-hyperkube.sh
#
# will build image localhost:5000/liyi/hyperkube-amd64:1.3.0-dev
 
set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/build/common.sh"

if [[ -z "${KUBE_DOCKER_REGISTRY:-}" ]]; then
	echo "KUBE_DOCKER_REGISTRY must be set"
	exit -1
fi
if [[ -z "${KUBE_DOCKER_OWNER:-}" ]]; then
	echo "KUBE_DOCKER_OWNER must be set"
	exit -1
fi
if [[ -z "${KUBE_DOCKER_VERSION:-}" ]]; then
	echo "KUBE_DOCKER_VERSION must be set"
	exit -1
fi

kube::build::verify_prereqs
kube::build::build_image
kube::build::run_build_command make WHAT=cmd/hyperkube

REGISTRY="${KUBE_DOCKER_REGISTRY}/${KUBE_DOCKER_OWNER}" \
VERSION="${KUBE_DOCKER_VERSION}" \
	make -C "${KUBE_ROOT}/cluster/images/hyperkube" build

docker push "${KUBE_DOCKER_REGISTRY}/${KUBE_DOCKER_OWNER}/hyperkube-amd64:${KUBE_DOCKER_VERSION}"
