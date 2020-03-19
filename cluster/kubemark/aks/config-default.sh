#!/usr/bin/env bash

# Copyright 2020 The Kubernetes Authors.
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

# shellcheck disable=SC2034 # Variables sourced in other scripts.

CLOUD_PROVIDER="${CLOUD_PROVIDER:-aks}"
CONTAINER_REGISTRY="${REGISTRY:-acr.io}"
KUBEMARK_IMAGE_REGISTRY="${CONTAINER_REGISTRY:-}"
KUBEMARK_IMAGE_MAKE_TARGET="${KUBEMARK_IMAGE_MAKE_TARGET:-gcloudpush}"

TEST_CLUSTER_LOG_LEVEL="${TEST_CLUSTER_LOG_LEVEL:---v=4}"
HOLLOW_KUBELET_TEST_LOG_LEVEL="${HOLLOW_KUBELET_TEST_LOG_LEVEL:-$TEST_CLUSTER_LOG_LEVEL}"
HOLLOW_PROXY_TEST_LOG_LEVEL="${HOLLOW_PROXY_TEST_LOG_LEVEL:-$TEST_CLUSTER_LOG_LEVEL}"

TEST_CLUSTER_API_CONTENT_TYPE="${TEST_CLUSTER_API_CONTENT_TYPE:-}"
KUBEMARK_KUBE_VERSION="${KUBEMARK_KUBE_VERSION:-1.17.0}"

# Hollow-node components' test arguments.
HOLLOW_KUBELET_TEST_ARGS="${HOLLOW_KUBELET_TEST_ARGS:-} ${HOLLOW_KUBELET_TEST_LOG_LEVEL}"
HOLLOW_PROXY_TEST_ARGS="${HOLLOW_PROXY_TEST_ARGS:-} ${HOLLOW_PROXY_TEST_LOG_LEVEL}"
# NUM_NODES is used by start-kubemark.sh to determine a correct number of replicas.
NUM_NODES=${KUBEMARK_NUM_NODES:-10}

KUBEMARK_NODE_SKU=${KUBEMARK_NODE_SKU:-Standard_D8s_v3}
KUBEMARK_RESOURCE_GROUP=${KUBEMARK_RESOURCE_GROUP:-}
KUBEMARK_RESOURCE_NAME=${KUBEMARK_RESOURCE_NAME:-}
KUBEMARK_LOCATION=${KUBEMARK_LOCATION:-southcentralus}
KUBEMARK_OS_DISK=${KUBEMARK_OS_DISK:-1024}
KUBEMARK_REAL_NODES=${KUBEMARK_REAL_NODES:-3}
USE_EXISTING=${USE_EXISTING:-false}
AZURE_CLIENT_ID="${AZURE_CLIENT_ID:-}"
AZURE_SUBSCRIPTION_ID="${AZURE_SUBSCRIPTION_ID:-}"
AZURE_TENANT_ID="${AZURE_TENANT_ID:-}"
AZURE_CLIENT_SECRET="${AZURE_CLIENT_SECRET:-}"
