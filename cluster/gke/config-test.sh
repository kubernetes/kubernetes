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

# The following are test-specific settings.
CLUSTER_NAME="${CLUSTER_NAME:-${USER}-gke-e2e}"
NETWORK=${KUBE_GKE_NETWORK:-e2e}
NODE_TAG="k8s-${CLUSTER_NAME}-node"
IMAGE_TYPE="${KUBE_GKE_IMAGE_TYPE:-}"


# For ease of maintenance, extract any pieces that do not vary between default
# and test in a common config.
source $(dirname "${BASH_SOURCE}")/config-common.sh
