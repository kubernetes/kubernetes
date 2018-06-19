#!/usr/bin/env bash

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

# Cloud information
CLOUD_PROVIDER="${CLOUD_PROVIDER:-iks}"
RANDGEN=$(dd if=/dev/urandom bs=64 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=16 count=1 2>/dev/null | sed 's/[A-Z]//g')
KUBE_NAMESPACE="kubemark_${RANDGEN}"
KUBEMARK_IMAGE_REGISTRY="${KUBEMARK_IMAGE_REGISTRY:-registry.ng.bluemix.net/${KUBE_NAMESPACE}}"
PROJECT_NAME="${PROJECT_NAME:-kubemark}"
KUBEMARK_IMAGE_TAG="${KUBEMARK_IMAGE_TAG:-3}"
KUBEMARK_IMAGE_LOCATION="${KUBEMARK_IMAGE_LOCATION:-${KUBE_ROOT}/cluster/images/kubemark}"
KUBEMARK_INIT_TAG="${KUBEMARK_INIT_TAG:-${PROJECT_NAME}:${KUBEMARK_IMAGE_TAG}}"
CLUSTER_LOCATION="${CLUSTER_LOCATION:-wdc06}"
REGISTRY_LOGIN_URL="${REGISTRY_LOGIN_URL:-https://api.ng.bluemix.net}"

# User defined
NUM_NODES="${NUM_NODES:-2}"
DESIRED_NODES="${DESIRED_NODES:-10}"
ENABLE_KUBEMARK_CLUSTER_AUTOSCALER="${ENABLE_KUBEMARK_CLUSTER_AUTOSCALER:-true}"
ENABLE_KUBEMARK_KUBE_DNS="${ENABLE_KUBEMARK_KUBE_DNS:-false}"
KUBELET_TEST_LOG_LEVEL="${KUBELET_TEST_LOG_LEVEL:-"--v=2"}"
KUBEPROXY_TEST_LOG_LEVEL="${KUBEPROXY_TEST_LOG_LEVEL:-"--v=4"}"
USE_REAL_PROXIER=${USE_REAL_PROXIER:-false}
NODE_INSTANCE_PREFIX=${NODE_INSTANCE_PREFIX:-node}
USE_EXISTING=${USE_EXISTING:-}

# Values for colored text
color_yellow='\033[1;33m'
color_norm='\033[0m'
color_blue='\033[1;34m'
color_red='\033[1;31m'
color_green='\033[1;32m'
color_cyan='\033[1;36m'
color_orange='\033[1;33m'
color_purple='\033[1;35m'


