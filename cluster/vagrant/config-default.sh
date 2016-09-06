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

## Contains configuration values for interacting with the Vagrant cluster

# Number of nodes in the cluster
NUM_NODES=${NUM_NODES-"1"}
export NUM_NODES

# The IP of the master
export MASTER_IP=${MASTER_IP-"10.245.1.2"}
export KUBE_MASTER_IP=${MASTER_IP}

export INSTANCE_PREFIX="kubernetes"
export MASTER_NAME="${INSTANCE_PREFIX}-master"

# Should the master serve as a node
REGISTER_MASTER_KUBELET=${REGISTER_MASTER:-false}

# Map out the IPs, names and container subnets of each node
export NODE_IP_BASE=${NODE_IP_BASE-"10.245.1."}
NODE_CONTAINER_SUBNET_BASE="10.246"
MASTER_CONTAINER_NETMASK="255.255.255.0"
MASTER_CONTAINER_ADDR="${NODE_CONTAINER_SUBNET_BASE}.0.1"
MASTER_CONTAINER_SUBNET="${NODE_CONTAINER_SUBNET_BASE}.0.1/24"
CONTAINER_SUBNET="${NODE_CONTAINER_SUBNET_BASE}.0.0/16"
for ((i=0; i < NUM_NODES; i++)) do
  NODE_IPS[$i]="${NODE_IP_BASE}$((i+3))"
  NODE_NAMES[$i]="${INSTANCE_PREFIX}-node-$((i+1))"
  NODE_CONTAINER_SUBNETS[$i]="${NODE_CONTAINER_SUBNET_BASE}.$((i+1)).1/24"
  NODE_CONTAINER_ADDRS[$i]="${NODE_CONTAINER_SUBNET_BASE}.$((i+1)).1"
  NODE_CONTAINER_NETMASKS[$i]="255.255.255.0"
  VAGRANT_NODE_NAMES[$i]="node-$((i+1))"
done

CLUSTER_IP_RANGE="${CLUSTER_IP_RANGE:-10.246.0.0/16}"

SERVICE_CLUSTER_IP_RANGE=10.247.0.0/16  # formerly PORTAL_NET

# Since this isn't exposed on the network, default to a simple user/passwd
MASTER_USER="${MASTER_USER:-vagrant}"
MASTER_PASSWD="${MASTER_PASSWD:-vagrant}"

# Admission Controllers to invoke prior to persisting objects in cluster
# If we included ResourceQuota, we should keep it at the end of the list to prevent incremeting quota usage prematurely.
ADMISSION_CONTROL=NamespaceLifecycle,LimitRanger,ServiceAccount,DefaultStorageClass,ResourceQuota

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

# Extra options to set on the Docker command line.  This is useful for setting
# --insecure-registry for local registries, or globally configuring selinux options
# TODO Enable selinux when Fedora 21 repositories get an updated docker package
#   see https://bugzilla.redhat.com/show_bug.cgi?id=1216151
#EXTRA_DOCKER_OPTS="${EXTRA_DOCKER_OPTS:-} -b=cbr0 --selinux-enabled --insecure-registry 10.0.0.0/8"
EXTRA_DOCKER_OPTS="${EXTRA_DOCKER_OPTS:-} --insecure-registry 10.0.0.0/8 -s overlay"

# Flag to tell the kubelet to enable CFS quota support
ENABLE_CPU_CFS_QUOTA="${KUBE_ENABLE_CPU_CFS_QUOTA:-true}"

# Optional: Install cluster DNS.
ENABLE_CLUSTER_DNS="${KUBE_ENABLE_CLUSTER_DNS:-true}"
DNS_SERVER_IP="10.247.0.10"
DNS_DOMAIN="cluster.local"
DNS_REPLICAS=1

# Optional: Install Kubernetes UI
ENABLE_CLUSTER_UI="${KUBE_ENABLE_CLUSTER_UI:-true}"

# Optional: Enable setting flags for kube-apiserver to turn on behavior in active-dev
RUNTIME_CONFIG="${KUBE_RUNTIME_CONFIG:-}"

# Determine extra certificate names for master
octets=($(echo "$SERVICE_CLUSTER_IP_RANGE" | sed -e 's|/.*||' -e 's/\./ /g'))
((octets[3]+=1))
service_ip=$(echo "${octets[*]}" | sed 's/ /./g')
MASTER_EXTRA_SANS="IP:${service_ip},DNS:kubernetes,DNS:kubernetes.default,DNS:kubernetes.default.svc,DNS:kubernetes.default.svc.${DNS_DOMAIN},DNS:${MASTER_NAME}"

NETWORK_PROVIDER="${NETWORK_PROVIDER:-none}" # opencontrail, kubenet, etc
if [ "${NETWORK_PROVIDER}" == "kubenet" ]; then
  CLUSTER_IP_RANGE="${CONTAINER_SUBNET}"
fi

# OpenContrail networking plugin specific settings
OPENCONTRAIL_TAG="${OPENCONTRAIL_TAG:-R2.20}"
OPENCONTRAIL_KUBERNETES_TAG="${OPENCONTRAIL_KUBERNETES_TAG:-master}"
OPENCONTRAIL_PUBLIC_SUBNET="${OPENCONTRAIL_PUBLIC_SUBNET:-10.1.0.0/16}"

# Optional: if set to true, kube-up will configure the cluster to run e2e tests.
E2E_STORAGE_TEST_ENVIRONMENT=${KUBE_E2E_STORAGE_TEST_ENVIRONMENT:-false}

# Default fallback NETWORK_IF_NAME, will be used in case when no 'VAGRANT-BEGIN' comments were defined in network-script
export DEFAULT_NETWORK_IF_NAME="eth0"

# Evict pods whenever compute resource availability on the nodes gets below a threshold.
EVICTION_HARD="${EVICTION_HARD:-memory.available<100Mi,nodefs.available<10%}"
