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

# Sane defaults for dev environments. The following variables can be easily overriden
# by setting each as a ENV variable ahead of time:
# KUBE_IMAGE, KUBE_MASTER_FLAVOR, KUBE_NODE_FLAVOR, NUM_NODES, NOVA_NETWORK and SSH_KEY_NAME

# Shared
KUBE_IMAGE="${KUBE_IMAGE-3eba4fbb-51da-4233-b699-8a4030561add}" # CoreOS (Stable)
SSH_KEY_NAME="${SSH_KEY_NAME-id_kubernetes}"
NOVA_NETWORK_LABEL="kubernetes-pool-net"
NOVA_NETWORK_CIDR="${NOVA_NETWORK-192.168.0.0/24}"
INSTANCE_PREFIX="kubernetes"

# Master
KUBE_MASTER_FLAVOR="${KUBE_MASTER_FLAVOR-general1-1}"
MASTER_NAME="${INSTANCE_PREFIX}-master"
MASTER_TAG="tags=${INSTANCE_PREFIX}-master"

# Node
KUBE_NODE_FLAVOR="${KUBE_NODE_FLAVOR-general1-2}"
NUM_NODES="${NUM_NODES-4}"
NODE_TAG="tags=${INSTANCE_PREFIX}-node"
NODE_NAMES=($(eval echo ${INSTANCE_PREFIX}-node-{1..${NUM_NODES}}))
KUBE_NETWORK="10.240.0.0/16"
SERVICE_CLUSTER_IP_RANGE="10.0.0.0/16"  # formerly PORTAL_NET

# Optional: Enable node logging.
ENABLE_NODE_LOGGING=false
LOGGING_DESTINATION=elasticsearch

# Optional: When set to true, Elasticsearch and Kibana will be setup as part of the cluster bring up.
ENABLE_CLUSTER_LOGGING=false
ELASTICSEARCH_LOGGING_REPLICAS=1

# Optional: Cluster monitoring to setup as part of the cluster bring up:
#   none     - No cluster monitoring setup
#   influxdb - Heapster, InfluxDB, and Grafana
#   google   - Heapster, Google Cloud Monitoring, and Google Cloud Logging
ENABLE_CLUSTER_MONITORING="${KUBE_ENABLE_CLUSTER_MONITORING:-influxdb}"

# Optional: Install cluster DNS.
ENABLE_CLUSTER_DNS="${KUBE_ENABLE_CLUSTER_DNS:-true}"
DNS_SERVER_IP="10.0.0.10"
DNS_DOMAIN="cluster.local"

# Optional: Enable DNS horizontal autoscaler
ENABLE_DNS_HORIZONTAL_AUTOSCALER="${KUBE_ENABLE_DNS_HORIZONTAL_AUTOSCALER:-false}"

# Optional: Install Kubernetes UI
ENABLE_CLUSTER_UI="${KUBE_ENABLE_CLUSTER_UI:-true}"
