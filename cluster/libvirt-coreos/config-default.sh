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

## Contains configuration values for interacting with the libvirt CoreOS cluster

# Number of minions in the cluster
NUM_MINIONS=${NUM_MINIONS:-3}
export NUM_MINIONS

# The IP of the master
export MASTER_IP="192.168.10.1"

export INSTANCE_PREFIX=kubernetes
export MASTER_NAME="${INSTANCE_PREFIX}-master"

# Map out the IPs, names and container subnets of each minion
export MINION_IP_BASE="192.168.10."
MINION_CONTAINER_SUBNET_BASE="10.10"
MASTER_CONTAINER_NETMASK="255.255.255.0"
MASTER_CONTAINER_ADDR="${MINION_CONTAINER_SUBNET_BASE}.0.1"
MASTER_CONTAINER_SUBNET="${MINION_CONTAINER_SUBNET_BASE}.0.1/24"
CONTAINER_SUBNET="${MINION_CONTAINER_SUBNET_BASE}.0.0/16"
if [[ "$NUM_MINIONS" -gt 253 ]]; then
  echo "ERROR: Because of how IPs are allocated in ${BASH_SOURCE}, you cannot create more than 253 minions"
  exit 1
fi
for ((i=0; i < NUM_MINIONS; i++)) do
  MINION_IPS[$i]="${MINION_IP_BASE}$((i+2))"
  MINION_NAMES[$i]="${INSTANCE_PREFIX}-minion-$((i+1))"
  MINION_CONTAINER_SUBNETS[$i]="${MINION_CONTAINER_SUBNET_BASE}.$((i+1)).1/24"
  MINION_CONTAINER_ADDRS[$i]="${MINION_CONTAINER_SUBNET_BASE}.$((i+1)).1"
  MINION_CONTAINER_NETMASKS[$i]="255.255.255.0"
done
MINION_CONTAINER_SUBNETS[$NUM_MINIONS]=$MASTER_CONTAINER_SUBNET

PORTAL_NET=10.11.0.0/16

# Optional: Install node monitoring.
ENABLE_NODE_MONITORING=true

# Optional: Enable node logging.
ENABLE_NODE_LOGGING=false
LOGGING_DESTINATION=elasticsearch

# Optional: Install cluster DNS.
ENABLE_CLUSTER_DNS=true
DNS_SERVER_IP="10.11.0.254"
DNS_DOMAIN="cluster.local"
DNS_REPLICAS=1
