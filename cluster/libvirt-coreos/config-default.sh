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

## Contains configuration values for interacting with the libvirt CoreOS cluster

# Number of minions in the cluster
NUM_NODES=${NUM_NODES:-3}
export NUM_NODES

# The IP of the master
export MASTER_IP="192.168.10.1"

export INSTANCE_PREFIX=kubernetes
export MASTER_NAME="${INSTANCE_PREFIX}-master"

# Map out the IPs, names and container subnets of each node
export NODE_IP_BASE="192.168.10."
NODE_CONTAINER_SUBNET_BASE="10.10"
MASTER_CONTAINER_NETMASK="255.255.255.0"
MASTER_CONTAINER_ADDR="${NODE_CONTAINER_SUBNET_BASE}.0.1"
MASTER_CONTAINER_SUBNET="${NODE_CONTAINER_SUBNET_BASE}.0.1/24"
CONTAINER_SUBNET="${NODE_CONTAINER_SUBNET_BASE}.0.0/16"
if [[ "$NUM_NODES" -gt 253 ]]; then
  echo "ERROR: Because of how IPs are allocated in ${BASH_SOURCE}, you cannot create more than 253 nodes"
  exit 1
fi
for ((i=0; i < NUM_NODES; i++)) do
  NODE_IPS[$i]="${NODE_IP_BASE}$((i+2))"
  NODE_NAMES[$i]="${INSTANCE_PREFIX}-node-$((i+1))"
  NODE_CONTAINER_SUBNETS[$i]="${NODE_CONTAINER_SUBNET_BASE}.$((i+1)).1/24"
  NODE_CONTAINER_ADDRS[$i]="${NODE_CONTAINER_SUBNET_BASE}.$((i+1)).1"
  NODE_CONTAINER_NETMASKS[$i]="255.255.255.0"
done
NODE_CONTAINER_SUBNETS[$NUM_NODES]=$MASTER_CONTAINER_SUBNET

SERVICE_CLUSTER_IP_RANGE="${SERVICE_CLUSTER_IP_RANGE:-10.11.0.0/16}"  # formerly PORTAL_NET


# Optional: Enable node logging.
ENABLE_NODE_LOGGING=false
LOGGING_DESTINATION=elasticsearch

# Optional: Install cluster DNS.
ENABLE_CLUSTER_DNS="${KUBE_ENABLE_CLUSTER_DNS:-true}"
DNS_SERVER_IP="${SERVICE_CLUSTER_IP_RANGE%.*}.254"
DNS_DOMAIN="cluster.local"

# Optional: Install cluster registry
ENABLE_CLUSTER_REGISTRY="${KUBE_ENABLE_CLUSTER_REGISTRY:-true}"

# Optional: Enable DNS horizontal autoscaler
ENABLE_DNS_HORIZONTAL_AUTOSCALER="${KUBE_ENABLE_DNS_HORIZONTAL_AUTOSCALER:-false}"

#Generate dns files
sed -f "${KUBE_ROOT}/cluster/addons/dns/transforms2sed.sed" < "${KUBE_ROOT}/cluster/addons/dns/kube-dns.yaml.base" | sed -f "${KUBE_ROOT}/cluster/libvirt-coreos/forShellEval.sed"  > "${KUBE_ROOT}/cluster/libvirt-coreos/kube-dns.yaml"


#Generate registry files
sed -f "${KUBE_ROOT}/cluster/libvirt-coreos/forEmptyDirRegistry.sed" < "${KUBE_ROOT}/cluster/addons/registry/registry-rc.yaml"  > "${KUBE_ROOT}/cluster/libvirt-coreos/registry-rc.yaml"
