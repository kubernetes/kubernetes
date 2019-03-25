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

# A configuration for Kubemark cluster. It doesn't need to be kept in
# sync with gce/config-default.sh (except the filename, because I'm reusing
# gce/util.sh script which assumes config filename), but if some things that
# are enabled by default should not run in hollow clusters, they should be disabled here.

# shellcheck disable=SC2034 # Variables sourced in other scripts.

source "${KUBE_ROOT}/cluster/gce/config-common.sh"

GCLOUD=gcloud
ZONE=${KUBE_GCE_ZONE:-us-central1-b}
REGION=${ZONE%-*}
NUM_NODES=${KUBEMARK_NUM_NODES:-10}
NUM_WINDOWS_NODES=${KUBEMARK_NUM_WINDOWS_NODES:-0}
MASTER_SIZE=${KUBEMARK_MASTER_SIZE:-n1-standard-$(get-master-size)}
MASTER_DISK_TYPE=pd-ssd
MASTER_DISK_SIZE=${MASTER_DISK_SIZE:-$(get-master-disk-size)}
MASTER_ROOT_DISK_SIZE=${KUBEMARK_MASTER_ROOT_DISK_SIZE:-$(get-master-root-disk-size)}
REGISTER_MASTER_KUBELET=${REGISTER_MASTER:-false}
PREEMPTIBLE_NODE=${PREEMPTIBLE_NODE:-false}
NODE_ACCELERATORS=${NODE_ACCELERATORS:-""}
CREATE_CUSTOM_NETWORK=${CREATE_CUSTOM_NETWORK:-false}
EVENT_PD=${EVENT_PD:-false}

MASTER_OS_DISTRIBUTION=${KUBE_MASTER_OS_DISTRIBUTION:-gci}
NODE_OS_DISTRIBUTION=${KUBE_NODE_OS_DISTRIBUTION:-gci}
MASTER_IMAGE=${KUBE_GCE_MASTER_IMAGE:-cos-beta-73-11647-64-0}
MASTER_IMAGE_PROJECT=${KUBE_GCE_MASTER_PROJECT:-cos-cloud}
CLEANUP_KUBEMARK_IMAGE=${CLEANUP_KUBEMARK_IMAGE:-true}

# GPUs supported in GCE do not have compatible drivers in Debian 7.
if [[ "${NODE_OS_DISTRIBUTION}" == "debian" ]]; then
    NODE_ACCELERATORS=""
fi

NETWORK=${KUBE_GCE_NETWORK:-e2e}
if [[ "${CREATE_CUSTOM_NETWORK}" == true ]]; then
  SUBNETWORK="${SUBNETWORK:-${NETWORK}-custom-subnet}"
fi
INSTANCE_PREFIX="${INSTANCE_PREFIX:-"default"}"
MASTER_NAME="${INSTANCE_PREFIX}-kubemark-master"
AGGREGATOR_MASTER_NAME="${INSTANCE_PREFIX}-kubemark-aggregator"
MASTER_TAG="kubemark-master"
EVENT_STORE_NAME="${INSTANCE_PREFIX}-event-store"
MASTER_IP_RANGE="${MASTER_IP_RANGE:-10.246.0.0/24}"
CLUSTER_IP_RANGE="${CLUSTER_IP_RANGE:-$(get-cluster-ip-range)}"
RUNTIME_CONFIG="${KUBE_RUNTIME_CONFIG:-}"
TERMINATED_POD_GC_THRESHOLD=${TERMINATED_POD_GC_THRESHOLD:-100}
KUBE_APISERVER_REQUEST_TIMEOUT=300
ETCD_COMPACTION_INTERVAL_SEC="${KUBEMARK_ETCD_COMPACTION_INTERVAL_SEC:-}"

# Set etcd image (e.g. k8s.gcr.io/etcd) and version (e.g. 3.1.12-1) if you need
# non-default version.
ETCD_IMAGE="${TEST_ETCD_IMAGE:-}"
ETCD_VERSION="${TEST_ETCD_VERSION:-}"
ETCD_SERVERS="${KUBEMARK_ETCD_SERVERS:-}"
ETCD_SERVERS_OVERRIDES="${KUBEMARK_ETCD_SERVERS_OVERRIDES:-}"

# Storage backend. 'etcd2' and 'etcd3' are supported.
STORAGE_BACKEND=${STORAGE_BACKEND:-}
# Storage media type: application/json and application/vnd.kubernetes.protobuf are supported.
STORAGE_MEDIA_TYPE=${STORAGE_MEDIA_TYPE:-}

# Default Log level for all components in test clusters and variables to override it in specific components.
TEST_CLUSTER_LOG_LEVEL="${TEST_CLUSTER_LOG_LEVEL:---v=4}"
API_SERVER_TEST_LOG_LEVEL="${API_SERVER_TEST_LOG_LEVEL:-$TEST_CLUSTER_LOG_LEVEL}"
CONTROLLER_MANAGER_TEST_LOG_LEVEL="${CONTROLLER_MANAGER_TEST_LOG_LEVEL:-$TEST_CLUSTER_LOG_LEVEL}"
SCHEDULER_TEST_LOG_LEVEL="${SCHEDULER_TEST_LOG_LEVEL:-$TEST_CLUSTER_LOG_LEVEL}"

HOLLOW_KUBELET_TEST_LOG_LEVEL="${HOLLOW_KUBELET_TEST_LOG_LEVEL:-$TEST_CLUSTER_LOG_LEVEL}"
HOLLOW_PROXY_TEST_LOG_LEVEL="${HOLLOW_PROXY_TEST_LOG_LEVEL:-$TEST_CLUSTER_LOG_LEVEL}"

TEST_CLUSTER_DELETE_COLLECTION_WORKERS="${TEST_CLUSTER_DELETE_COLLECTION_WORKERS:---delete-collection-workers=16}"
TEST_CLUSTER_MAX_REQUESTS_INFLIGHT="${TEST_CLUSTER_MAX_REQUESTS_INFLIGHT:-}"
TEST_CLUSTER_RESYNC_PERIOD="${TEST_CLUSTER_RESYNC_PERIOD:-}"

# ContentType used by all components to communicate with apiserver.
TEST_CLUSTER_API_CONTENT_TYPE="${TEST_CLUSTER_API_CONTENT_TYPE:-}"

KUBEMARK_MASTER_COMPONENTS_QPS_LIMITS="${KUBEMARK_MASTER_COMPONENTS_QPS_LIMITS:-}"

CUSTOM_ADMISSION_PLUGINS="${CUSTOM_ADMISSION_PLUGINS:-NamespaceLifecycle,LimitRanger,ServiceAccount,PersistentVolumeLabel,DefaultStorageClass,DefaultTolerationSeconds,NodeRestriction,Priority,StorageObjectInUseProtection,MutatingAdmissionWebhook,ValidatingAdmissionWebhook,ResourceQuota}"

# Master components' test arguments.
APISERVER_TEST_ARGS="${KUBEMARK_APISERVER_TEST_ARGS:-} --runtime-config=extensions/v1beta1,scheduling.k8s.io/v1alpha1 ${API_SERVER_TEST_LOG_LEVEL} ${TEST_CLUSTER_MAX_REQUESTS_INFLIGHT} ${TEST_CLUSTER_DELETE_COLLECTION_WORKERS}"
CONTROLLER_MANAGER_TEST_ARGS="${KUBEMARK_CONTROLLER_MANAGER_TEST_ARGS:-} ${CONTROLLER_MANAGER_TEST_LOG_LEVEL} ${TEST_CLUSTER_RESYNC_PERIOD} ${TEST_CLUSTER_API_CONTENT_TYPE} ${KUBEMARK_MASTER_COMPONENTS_QPS_LIMITS}"
SCHEDULER_TEST_ARGS="${KUBEMARK_SCHEDULER_TEST_ARGS:-} ${SCHEDULER_TEST_LOG_LEVEL} ${TEST_CLUSTER_API_CONTENT_TYPE} ${KUBEMARK_MASTER_COMPONENTS_QPS_LIMITS}"

# Hollow-node components' test arguments.
HOLLOW_KUBELET_TEST_ARGS="${HOLLOW_KUBELET_TEST_ARGS:-} ${HOLLOW_KUBELET_TEST_LOG_LEVEL}"
HOLLOW_PROXY_TEST_ARGS="${HOLLOW_PROXY_TEST_ARGS:-} ${HOLLOW_PROXY_TEST_LOG_LEVEL}"

SERVICE_CLUSTER_IP_RANGE="10.0.0.0/16"  # formerly PORTAL_NET
ALLOCATE_NODE_CIDRS=true

# Optional: Enable cluster autoscaler.
ENABLE_KUBEMARK_CLUSTER_AUTOSCALER="${ENABLE_KUBEMARK_CLUSTER_AUTOSCALER:-false}"
# When using Cluster Autoscaler, always start with one hollow-node replica.
# NUM_NODES should not be specified by the user. Instead we use
# NUM_NODES=KUBEMARK_AUTOSCALER_MAX_NODES. This gives other cluster components
# (e.g. kubemark master, Heapster) enough resources to handle maximum cluster size.
if [[ "${ENABLE_KUBEMARK_CLUSTER_AUTOSCALER}" == "true" ]]; then
  NUM_REPLICAS=1
  if [[ -n "$NUM_NODES" ]]; then
    echo "WARNING: Using Cluster Autoscaler, ignoring NUM_NODES parameter. Set KUBEMARK_AUTOSCALER_MAX_NODES to specify maximum size of the cluster."
  fi
fi

#Optional: Enable kube dns.
ENABLE_KUBEMARK_KUBE_DNS="${ENABLE_KUBEMARK_KUBE_DNS:-true}"
KUBE_DNS_DOMAIN="${KUBE_DNS_DOMAIN:-cluster.local}"

# Optional: set feature gates
FEATURE_GATES="${KUBE_FEATURE_GATES:-ExperimentalCriticalPodAnnotation=true}"

# Enable a simple "AdvancedAuditing" setup for testing.
ENABLE_APISERVER_ADVANCED_AUDIT="${ENABLE_APISERVER_ADVANCED_AUDIT:-false}"

# Optional: enable pod priority
ENABLE_POD_PRIORITY="${ENABLE_POD_PRIORITY:-}"
if [[ "${ENABLE_POD_PRIORITY}" == "true" ]]; then
    FEATURE_GATES="${FEATURE_GATES},PodPriority=true"
fi

# The number of services that are allowed to sync concurrently. Will be passed
# into kube-controller-manager via `--concurrent-service-syncs`
CONCURRENT_SERVICE_SYNCS="${CONCURRENT_SERVICE_SYNCS:-}"
