#!/bin/bash

# Copyright 2015 The Kubernetes Authors All rights reserved.
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

# A configuration for Kubemark cluster. It doesn't need to be kept in
# sync with gce/config-default.sh (except the filename, because I'm reusing
# gce/util.sh script which assumes config filename), but if some things that
# are enabled by default should not run in hollow clusters, they should be disabled here.

GCLOUD=gcloud
ZONE=${KUBE_GCE_ZONE:-us-central1-b}
MASTER_SIZE=${MASTER_SIZE:-n1-standard-4}
NUM_MINIONS=${NUM_MINIONS:-100}
MASTER_DISK_TYPE=pd-ssd
MASTER_DISK_SIZE=${MASTER_DISK_SIZE:-20GB}
REGISTER_MASTER_KUBELET=${REGISTER_MASTER:-false}
PREEMPTIBLE_MINION=${PREEMPTIBLE_MINION:-false}

OS_DISTRIBUTION=${KUBE_OS_DISTRIBUTION:-debian}
MASTER_IMAGE=${KUBE_GCE_MASTER_IMAGE:-container-vm-v20150806}
MASTER_IMAGE_PROJECT=${KUBE_GCE_MASTER_PROJECT:-google-containers}

NETWORK=${KUBE_GCE_NETWORK:-default}
INSTANCE_PREFIX="${INSTANCE_PREFIX:-"default"}"
MASTER_NAME="${INSTANCE_PREFIX}-kubemark-master"
MASTER_TAG="kubemark-master"
MASTER_IP_RANGE="${MASTER_IP_RANGE:-10.246.0.0/24}"
CLUSTER_IP_RANGE="${CLUSTER_IP_RANGE:-10.244.0.0/16}"
RUNTIME_CONFIG="${KUBE_RUNTIME_CONFIG:-}"

# Increase the sleep interval value if concerned about API rate limits. 3, in seconds, is the default.
POLL_SLEEP_INTERVAL=3
SERVICE_CLUSTER_IP_RANGE="10.0.0.0/16"  # formerly PORTAL_NET
ALLOCATE_NODE_CIDRS=true

ENABLE_CLUSTER_MONITORING="${KUBE_ENABLE_CLUSTER_MONITORING:-none}"
ENABLE_NODE_LOGGING="${KUBE_ENABLE_NODE_LOGGING:-false}"
ENABLE_CLUSTER_LOGGING="${KUBE_ENABLE_CLUSTER_LOGGING:-false}"
# Optional: Don't require https for registries in our local RFC1918 network
if [[ ${KUBE_ENABLE_INSECURE_REGISTRY:-false} == "true" ]]; then
  EXTRA_DOCKER_OPTS="--insecure-registry 10.0.0.0/8"
fi

ENABLE_CLUSTER_DNS="${KUBE_ENABLE_CLUSTER_DNS:-false}"
ENABLE_CLUSTER_REGISTRY="${KUBE_ENABLE_CLUSTER_REGISTRY:-false}"
ENABLE_CLUSTER_UI="${KUBE_ENABLE_CLUSTER_UI:-false}"
ENABLE_NODE_AUTOSCALER="${KUBE_ENABLE_NODE_AUTOSCALER:-false}"

# Admission Controllers to invoke prior to persisting objects in cluster
ADMISSION_CONTROL=NamespaceLifecycle,LimitRanger,SecurityContextDeny,ServiceAccount,ResourceQuota

# Optional: if set to true kube-up will automatically check for existing resources and clean them up.
KUBE_UP_AUTOMATIC_CLEANUP=${KUBE_UP_AUTOMATIC_CLEANUP:-false}
