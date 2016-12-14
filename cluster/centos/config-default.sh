#!/bin/bash

# Copyright 2015 The Kubernetes Authors.
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

## Contains configuration values for the CentOS cluster
# The user should have sudo privilege
export MASTER=${MASTER:-"centos@172.10.0.11"}
export MASTER_IP=${MASTER#*@}

# Define all your minion nodes,
# And separated with blank space like <user_1@ip_1> <user_2@ip_2> <user_3@ip_3>.
# The user should have sudo privilege
export NODES=${NODES:-"centos@172.10.0.12 centos@172.10.0.13"}

# Number of nodes in your cluster.
export NUM_NODES=${NUM_NODES:-2}

# Should be removed when NUM_NODES is deprecated in validate-cluster.sh
export NUM_NODES=${NUM_NODES}

# By default, the cluster will use the etcd installed on master.
export ETCD_SERVERS=${ETCD_SERVERS:-"http://$MASTER_IP:2379"}

# define the IP range used for service cluster IPs.
# according to rfc 1918 ref: https://tools.ietf.org/html/rfc1918 choose a private ip range here.
export SERVICE_CLUSTER_IP_RANGE=${SERVICE_CLUSTER_IP_RANGE:-"192.168.3.0/24"}

# Optional: Install cluster DNS.
ENABLE_CLUSTER_DNS="${KUBE_ENABLE_CLUSTER_DNS:-true}"
# DNS_SERVER_IP must be a IP in SERVICE_CLUSTER_IP_RANGE
DNS_SERVER_IP=${DNS_SERVER_IP:-"192.168.3.100"}
DNS_DOMAIN=${DNS_DOMAIN:-"cluster.local"}

# Optional: Install Kubernetes UI
ENABLE_CLUSTER_UI="${KUBE_ENABLE_CLUSTER_UI:-true}"

# define the IP range used for flannel overlay network, should not conflict with above SERVICE_CLUSTER_IP_RANGE
export FLANNEL_NET=${FLANNEL_NET:-"172.16.0.0/16"}

# Admission Controllers to invoke prior to persisting objects in cluster
# If we included ResourceQuota, we should keep it at the end of the list to prevent incrementing quota usage prematurely.
export ADMISSION_CONTROL=NamespaceLifecycle,LimitRanger,ServiceAccount,ResourceQuota

# Extra options to set on the Docker command line.
# This is useful for setting --insecure-registry for local registries.
export DOCKER_OPTS=${DOCKER_OPTS:-""} 


# Timeouts for process checking on master and minion
export PROCESS_CHECK_TIMEOUT=${PROCESS_CHECK_TIMEOUT:-180} # seconds.
