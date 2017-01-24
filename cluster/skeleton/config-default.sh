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

source "${KUBE_ROOT}/cluster/common.sh"
source "${KUBE_ROOT}/cluster/skeleton/util.sh"
source "${KUBE_ROOT}/cluster/lib/util.sh"

# Use an existing Kubernetes cluster at $MASTER_IP
MASTER_IP=""
MASTER_NAME="${MASTER_NAME:-$MASTER_IP}"
SERVICE_CLUSTER_IP_RANGE="${SERVICE_CLUSTER_IP_RANGE:-$MASTER_IP}"
PROJECT="${PROJECT:-kubemark}"

VM_USER="${VM_USER:-kubernetes}"
COPY_DIR="${COPY_DIR:-/home/$VM_USER}"

# Default number of retries when executing a command
RETRIES="${RETRIES:-3}"
