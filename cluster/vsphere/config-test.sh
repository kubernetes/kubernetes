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

NUM_NODES=2
DISK=./kube/kube.vmdk
GUEST_ID=debian7_64Guest

INSTANCE_PREFIX="e2e-test-${USER}"
MASTER_TAG="${INSTANCE_PREFIX}-master"
NODE_TAG="${INSTANCE_PREFIX}-minion"

MASTER_NAME="${INSTANCE_PREFIX}-master"
MASTER_MEMORY_MB=1024
MASTER_CPU=1

NODE_NAMES=($(eval echo ${INSTANCE_PREFIX}-minion-{1..${NUM_NODES}}))
NODE_IP_RANGES="10.244.0.0/16"
MASTER_IP_RANGE="${MASTER_IP_RANGE:-10.246.0.0/24}"
NODE_MEMORY_MB=1024
NODE_CPU=1

SERVICE_CLUSTER_IP_RANGE="10.244.240.0/20"  # formerly PORTAL_NET

# Optional: if set to true, kube-up will configure the cluster to run e2e tests.
E2E_STORAGE_TEST_ENVIRONMENT=${KUBE_E2E_STORAGE_TEST_ENVIRONMENT:-false}
