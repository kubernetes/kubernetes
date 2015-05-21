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

INSTANCE_PREFIX=kubernetes
AZ_LOCATION='West US'
TAG=testing
AZ_CS_PREFIX=kube
AZ_VNET=MyVnet
AZ_SUBNET=Subnet-1
AZ_IMAGE=b39f27a8b8c64d52b05eac6a62ebad85__Ubuntu-14_04_1-LTS-amd64-server-20140927-en-us-30GB
AZ_CS="" # is set in azure/util.sh verify-prereqs

AZ_SSH_KEY=$HOME/.ssh/azure_rsa
AZ_SSH_CERT=$HOME/.ssh/azure.pem

NUM_MINIONS=4

MASTER_NAME="${INSTANCE_PREFIX}-master"
MASTER_TAG="${INSTANCE_PREFIX}-master"
MINION_TAG="${INSTANCE_PREFIX}-minion"
MINION_NAMES=($(eval echo ${INSTANCE_PREFIX}-minion-{1..${NUM_MINIONS}}))
MINION_IP_RANGES=($(eval echo "10.244.{1..${NUM_MINIONS}}.0/24"))
MINION_SCOPES=""

PORTAL_NET="10.250.0.0/16"

# Optional: Install node logging
ENABLE_NODE_LOGGING=false
LOGGING_DESTINATION=elasticsearch # options: elasticsearch, gcp

# Optional: When set to true, Elasticsearch and Kibana will be setup as part of the cluster bring up.
ENABLE_CLUSTER_LOGGING=false
ELASTICSEARCH_LOGGING_REPLICAS=1

# Optional: When set to true, heapster, Influxdb and grafana will be setup as part of the cluster bring up.
ENABLE_CLUSTER_MONITORING="${KUBE_ENABLE_CLUSTER_MONITORING:-true}"

# Admission Controllers to invoke prior to persisting objects in cluster
ADMISSION_CONTROL=NamespaceLifecycle,NamespaceAutoProvision,LimitRanger,SecurityContextDeny,ServiceAccount,ResourceQuota
