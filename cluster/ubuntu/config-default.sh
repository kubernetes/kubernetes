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

## Contains configuration values for the Ubuntu cluster

# Define all your cluster nodes, MASTER node comes first"
# And separated with blank space like <user_1@ip_1> <user_2@ip_2> <user_3@ip_3> 
export nodes=${nodes:-"vcap@10.10.103.250 vcap@10.10.103.162 vcap@10.10.103.223"}

# Define all your nodes role: a(master) or i(minion) or ai(both master and minion), must be the order same 
role=${role:-"ai i i"}
# If it practically impossible to set an array as an environment variable
# from a script, so assume variable is a string then convert it to an array
export roles=($role)

# Define minion numbers
export NUM_MINIONS=${NUM_MINIONS:-3}
# define the IP range used for service cluster IPs.
# according to rfc 1918 ref: https://tools.ietf.org/html/rfc1918 choose a private ip range here.
export SERVICE_CLUSTER_IP_RANGE=${SERVICE_CLUSTER_IP_RANGE:-192.168.3.0/24}  # formerly PORTAL_NET
# define the IP range used for flannel overlay network, should not conflict with above SERVICE_CLUSTER_IP_RANGE
export FLANNEL_NET=${FLANNEL_NET:-172.16.0.0/16}

# Admission Controllers to invoke prior to persisting objects in cluster
export ADMISSION_CONTROL=NamespaceLifecycle,LimitRanger,ServiceAccount,ResourceQuota,DenyEscalatingExec,SecurityContextDeny

SERVICE_NODE_PORT_RANGE=${SERVICE_NODE_PORT_RANGE:-"30000-32767"}

# Optional: Enable node logging.
ENABLE_NODE_LOGGING=false
LOGGING_DESTINATION=${LOGGING_DESTINATION:-elasticsearch}

# Optional: When set to true, Elasticsearch and Kibana will be setup as part of the cluster bring up.
ENABLE_CLUSTER_LOGGING=false
ELASTICSEARCH_LOGGING_REPLICAS=${ELASTICSEARCH_LOGGING_REPLICAS:-1}

# Optional: When set to true, heapster, Influxdb and Grafana will be setup as part of the cluster bring up.
ENABLE_CLUSTER_MONITORING="${KUBE_ENABLE_CLUSTER_MONITORING:-true}"

# Extra options to set on the Docker command line.  This is useful for setting
# --insecure-registry for local registries.
DOCKER_OPTS=${DOCKER_OPTS:-""}

# Optional: Install cluster DNS.
ENABLE_CLUSTER_DNS="${KUBE_ENABLE_CLUSTER_DNS:-true}"
# DNS_SERVER_IP must be a IP in SERVICE_CLUSTER_IP_RANGE
DNS_SERVER_IP=${DNS_SERVER_IP:-"192.168.3.10"}
DNS_DOMAIN=${DNS_DOMAIN:-"cluster.local"}
DNS_REPLICAS=${DNS_REPLICAS:-1}

# Optional: Install Kubernetes UI
ENABLE_CLUSTER_UI="${KUBE_ENABLE_CLUSTER_UI:-true}"

# Optional: Enable setting flags for kube-apiserver to turn on behavior in active-dev
#RUNTIME_CONFIG=""

# Optional: Add http or https proxy when download easy-rsa.
# Add envitonment variable separated with blank space like "http_proxy=http://10.x.x.x:8080 https_proxy=https://10.x.x.x:8443"
PROXY_SETTING=${PROXY_SETTING:-""}

