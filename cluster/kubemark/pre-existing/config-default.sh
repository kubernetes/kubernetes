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

# Configuration for landing a Kubemark cluster on a pre-existing Kubernetes
# cluster.

# Pre-existing cluster expects a MASTER_IP
MASTER_IP="${MASTER_IP:-}"
MASTER_IP_AND_PORT="${MASTER_IP_AND_PORT:-$MASTER_IP:6443}"

# $PROJECT/$CONTAINER_REGISTRY/kubemark
PROJECT="${PROJECT:-}"
CONTAINER_REGISTRY="${CONTAINER_REGISTRY:-}"

NUM_NODES="${NUM_NODES:-1}"
SERVICE_CLUSTER_IP_RANGE="${SERVICE_CLUSTER_IP_RANGE:-}"
TEST_CLUSTER_API_CONTENT_TYPE="${TEST_CLUSTER_API_CONTENT_TYPE:-}"
KUBELET_TEST_LOG_LEVEL="${KUBELET_TEST_LOG_LEVEL:-}"
KUBEPROXY_TEST_LOG_LEVEL="${KUBEPROXY_TEST_LOG_LEVEL:-}"

MASTER_NAME="${MASTER_NAME:-}"
DNS_DOMAIN="${DNS_DOMAIN:-cluster.local}"
