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

## Contains configuration values for interacting with the Vagrant cluster

# Number of minions in the cluster
NUM_MINIONS=${NUM_MINIONS-"1"}
export NUM_MINIONS

# The IP of the master
export MASTER_IP="10.245.1.2"
export KUBE_MASTER_IP="10.245.1.2"

export INSTANCE_PREFIX="kubernetes"
export MASTER_NAME="${INSTANCE_PREFIX}-master"

# Map out the IPs, names and container subnets of each minion
export MINION_IP_BASE="10.245.1."
MINION_CONTAINER_SUBNET_BASE="10.246"
MASTER_CONTAINER_NETMASK="255.255.255.0"
MASTER_CONTAINER_ADDR="${MINION_CONTAINER_SUBNET_BASE}.0.1"
MASTER_CONTAINER_SUBNET="${MINION_CONTAINER_SUBNET_BASE}.0.1/24"
CONTAINER_SUBNET="${MINION_CONTAINER_SUBNET_BASE}.0.0/16"
for ((i=0; i < NUM_MINIONS; i++)) do
  MINION_IPS[$i]="${MINION_IP_BASE}$((i+3))"
  MINION_NAMES[$i]="${INSTANCE_PREFIX}-minion-$((i+1))"
  MINION_CONTAINER_SUBNETS[$i]="${MINION_CONTAINER_SUBNET_BASE}.$((i+1)).1/24"
  MINION_CONTAINER_ADDRS[$i]="${MINION_CONTAINER_SUBNET_BASE}.$((i+1)).1"
  MINION_CONTAINER_NETMASKS[$i]="255.255.255.0"
  VAGRANT_MINION_NAMES[$i]="minion-$((i+1))"
done

PORTAL_NET=10.247.0.0/16

# Since this isn't exposed on the network, default to a simple user/passwd
MASTER_USER=vagrant
MASTER_PASSWD=vagrant

# Admission Controllers to invoke prior to persisting objects in cluster
ADMISSION_CONTROL=NamespaceLifecycle,NamespaceAutoProvision,LimitRanger,SecurityContextDeny,ServiceAccount,ResourceQuota

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

# Extra options to set on the Docker command line.  This is useful for setting
# --insecure-registry for local registries.
DOCKER_OPTS=""

# Optional: Install cluster DNS.
ENABLE_CLUSTER_DNS=true
DNS_SERVER_IP="10.247.0.10"
DNS_DOMAIN="kubernetes.local"
DNS_REPLICAS=1

# Optional: Enable setting flags for kube-apiserver to turn on behavior in active-dev
#RUNTIME_CONFIG=""
RUNTIME_CONFIG="api/v1beta3"
