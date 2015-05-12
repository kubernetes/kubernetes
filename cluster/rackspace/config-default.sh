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

# Sane defaults for dev environments. The following variables can be easily overriden
# by setting each as a ENV variable ahead of time:
# KUBE_IMAGE, KUBE_MASTER_FLAVOR, KUBE_MINION_FLAVOR, NUM_MINIONS, NOVA_NETWORK and SSH_KEY_NAME

# Shared
KUBE_IMAGE="${KUBE_IMAGE-b63e1435-a46f-4726-b984-e3f15ae92753}" # CoreOS(Beta) 
SSH_KEY_NAME="${SSH_KEY_NAME-id_kubernetes}"
NOVA_NETWORK_LABEL="kubernetes-pool-net"
NOVA_NETWORK_CIDR="${NOVA_NETWORK-192.168.0.0/24}"
INSTANCE_PREFIX="kubernetes"

# Master
KUBE_MASTER_FLAVOR="${KUBE_MASTER_FLAVOR-performance1-1}"
MASTER_NAME="${INSTANCE_PREFIX}-master"
MASTER_TAG="tags=${INSTANCE_PREFIX}-master"

# Minion
KUBE_MINION_FLAVOR="${KUBE_MINION_FLAVOR-performance1-2}"
RAX_NUM_MINIONS="${RAX_NUM_MINIONS-4}"
MINION_TAG="tags=${INSTANCE_PREFIX}-minion"
MINION_NAMES=($(eval echo ${INSTANCE_PREFIX}-minion-{1..${RAX_NUM_MINIONS}}))
KUBE_NETWORK=($(eval echo "10.240.{1..${RAX_NUM_MINIONS}}.0/24"))
PORTAL_NET="10.0.0.0/16"

# Optional: Install node monitoring.
ENABLE_NODE_MONITORING=true

# Optional: Enable node logging.
ENABLE_NODE_LOGGING=false
LOGGING_DESTINATION=elasticsearch

# Optional: When set to true, Elasticsearch and Kibana will be setup as part of the cluster bring up.
ENABLE_CLUSTER_LOGGING=false
ELASTICSEARCH_LOGGING_REPLICAS=1

# Optional: When set to true, heapster, Influxdb and Grafana will be setup as part of the cluster bring up.
ENABLE_CLUSTER_MONITORING="${KUBE_ENABLE_CLUSTER_MONITORING:-true}"

# Optional: Install cluster DNS.
ENABLE_CLUSTER_DNS=true
DNS_SERVER_IP="10.0.0.10"
DNS_DOMAIN="cluster.local"
DNS_REPLICAS=1
