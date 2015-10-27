#!/bin/bash

# Copyright 2014 The Kubernetes Authors All rights reserved.
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

# TODO(jbeda): Provide a way to override project
# gcloud multiplexing for shared GCE/GKE tests.
GCLOUD=gcloud
ZONE=${KUBE_GCE_ZONE:-us-central1-b}
MASTER_SIZE=${MASTER_SIZE:-n1-standard-1}
MINION_SIZE=${MINION_SIZE:-n1-standard-1}
NUM_MINIONS=${NUM_MINIONS:-4}
MASTER_DISK_TYPE=pd-ssd
MASTER_DISK_SIZE=${MASTER_DISK_SIZE:-20GB}
MINION_DISK_TYPE=pd-standard
MINION_DISK_SIZE=${MINION_DISK_SIZE:-100GB}
REGISTER_MASTER_KUBELET=${REGISTER_MASTER:-false}
PREEMPTIBLE_MINION=${PREEMPTIBLE_MINION:-false}

OS_DISTRIBUTION=${KUBE_OS_DISTRIBUTION:-debian}
MASTER_IMAGE=${KUBE_GCE_MASTER_IMAGE:-container-vm-v20150806}
MASTER_IMAGE_PROJECT=${KUBE_GCE_MASTER_PROJECT:-google-containers}
MINION_IMAGE=${KUBE_GCE_MINION_IMAGE:-"${MASTER_IMAGE}"}
MINION_IMAGE_PROJECT=${KUBE_GCE_MINION_PROJECT:-"${MASTER_IMAGE_PROJECT}"}
CONTAINER_RUNTIME=${KUBE_CONTAINER_RUNTIME:-docker}
RKT_VERSION=${KUBE_RKT_VERSION:-0.5.5}

NETWORK=${KUBE_GCE_NETWORK:-default}
INSTANCE_PREFIX="${KUBE_GCE_INSTANCE_PREFIX:-kubernetes}"
MASTER_NAME="${INSTANCE_PREFIX}-master"
MASTER_TAG="${INSTANCE_PREFIX}-master"
MINION_TAG="${INSTANCE_PREFIX}-minion"
MASTER_IP_RANGE="${MASTER_IP_RANGE:-10.246.0.0/24}"
CLUSTER_IP_RANGE="${CLUSTER_IP_RANGE:-10.244.0.0/16}"
MINION_SCOPES="${MINION_SCOPES:-compute-rw,monitoring,logging-write,storage-ro}"
RUNTIME_CONFIG="${KUBE_RUNTIME_CONFIG:-}"
ENABLE_EXPERIMENTAL_API="${KUBE_ENABLE_EXPERIMENTAL_API:-false}"

# Increase the sleep interval value if concerned about API rate limits. 3, in seconds, is the default.
POLL_SLEEP_INTERVAL=3
SERVICE_CLUSTER_IP_RANGE="10.0.0.0/16"  # formerly PORTAL_NET
ALLOCATE_NODE_CIDRS=true

# Optional: Cluster monitoring to setup as part of the cluster bring up:
#   none           - No cluster monitoring setup
#   influxdb       - Heapster, InfluxDB, and Grafana
#   google         - Heapster, Google Cloud Monitoring, and Google Cloud Logging
#   googleinfluxdb - Enable influxdb and google (except GCM)
#   standalone     - Heapster only. Metrics available via Heapster REST API.
ENABLE_CLUSTER_MONITORING="${KUBE_ENABLE_CLUSTER_MONITORING:-googleinfluxdb}"

# Optional: Enable node logging.
ENABLE_NODE_LOGGING="${KUBE_ENABLE_NODE_LOGGING:-true}"
LOGGING_DESTINATION="${KUBE_LOGGING_DESTINATION:-gcp}" # options: elasticsearch, gcp

# Optional: When set to true, Elasticsearch and Kibana will be setup as part of the cluster bring up.
ENABLE_CLUSTER_LOGGING="${KUBE_ENABLE_CLUSTER_LOGGING:-true}"
ELASTICSEARCH_LOGGING_REPLICAS=1

# Optional: Don't require https for registries in our local RFC1918 network
if [[ ${KUBE_ENABLE_INSECURE_REGISTRY:-false} == "true" ]]; then
  EXTRA_DOCKER_OPTS="--insecure-registry 10.0.0.0/8"
fi

# Optional: Install cluster DNS.
ENABLE_CLUSTER_DNS="${KUBE_ENABLE_CLUSTER_DNS:-true}"
DNS_SERVER_IP="10.0.0.10"
DNS_DOMAIN="cluster.local"
DNS_REPLICAS=1

# Optional: Install cluster docker registry.
ENABLE_CLUSTER_REGISTRY="${KUBE_ENABLE_CLUSTER_REGISTRY:-true}"
CLUSTER_REGISTRY_DISK="${CLUSTER_REGISTRY_PD:-${INSTANCE_PREFIX}-kube-system-kube-registry}"
CLUSTER_REGISTRY_DISK_SIZE="${CLUSTER_REGISTRY_DISK_SIZE:-200GB}"
CLUSTER_REGISTRY_DISK_TYPE_GCE="${CLUSTER_REGISTRY_DISK_TYPE_GCE:-pd-standard}"

# Optional: Install Kubernetes UI
ENABLE_CLUSTER_UI="${KUBE_ENABLE_CLUSTER_UI:-true}"

# Optional: Create autoscaler for cluster's nodes.
ENABLE_NODE_AUTOSCALER="${KUBE_ENABLE_NODE_AUTOSCALER:-false}"
if [[ "${ENABLE_NODE_AUTOSCALER}" == "true" ]]; then
  AUTOSCALER_MIN_NODES="${KUBE_AUTOSCALER_MIN_NODES:-1}"
  AUTOSCALER_MAX_NODES="${KUBE_AUTOSCALER_MAX_NODES:-${NUM_MINIONS}}"
  TARGET_NODE_UTILIZATION="${KUBE_TARGET_NODE_UTILIZATION:-0.7}"
fi

# Optional: Enable feature for autoscaling number of pods
# Experimental feature, not ready for production use.
ENABLE_HORIZONTAL_POD_AUTOSCALER="${KUBE_ENABLE_HORIZONTAL_POD_AUTOSCALER:-false}"
if [[ "${ENABLE_HORIZONTAL_POD_AUTOSCALER}" == "true" ]]; then
  ENABLE_EXPERIMENTAL_API=true
fi

# Optional: Enable deployment experimental feature, not ready for production use.
ENABLE_DEPLOYMENTS="${KUBE_ENABLE_DEPLOYMENTS:-false}"
if [[ "${ENABLE_DEPLOYMENTS}" == "true" ]]; then
  ENABLE_EXPERIMENTAL_API=true
fi
# Optional: Enable daemonset experimental feature, not ready for production use.
ENABLE_DAEMONSETS="${KUBE_ENABLE_DAEMONSETS:-false}"
if [[ "${ENABLE_DAEMONSETS}" == "true" ]]; then
  ENABLE_EXPERIMENTAL_API=true
fi

# Admission Controllers to invoke prior to persisting objects in cluster
ADMISSION_CONTROL=NamespaceLifecycle,LimitRanger,SecurityContextDeny,ServiceAccount,ResourceQuota

# Optional: if set to true kube-up will automatically check for existing resources and clean them up.
KUBE_UP_AUTOMATIC_CLEANUP=${KUBE_UP_AUTOMATIC_CLEANUP:-false}
