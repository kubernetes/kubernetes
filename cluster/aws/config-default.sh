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

ZONE=${KUBE_AWS_ZONE:-us-west-2a}
MASTER_SIZE=${MASTER_SIZE:-t2.micro}
MINION_SIZE=${MINION_SIZE:-t2.micro}
NUM_MINIONS=${NUM_MINIONS:-4}

# Optional: Set AWS_S3_BUCKET to the name of an S3 bucket to use for uploading binaries
# (otherwise a unique bucket name will be generated for you)
#  AWS_S3_BUCKET=kubernetes-artifacts

# Because regions are globally named, we want to create in a single region; default to us-east-1
AWS_S3_REGION=${AWS_S3_REGION:-us-east-1}

INSTANCE_PREFIX="${KUBE_AWS_INSTANCE_PREFIX:-kubernetes}"
CLUSTER_ID=${INSTANCE_PREFIX}
AWS_SSH_KEY=${AWS_SSH_KEY:-$HOME/.ssh/kube_aws_rsa}
IAM_PROFILE_MASTER="kubernetes-master"
IAM_PROFILE_MINION="kubernetes-minion"

LOG="/dev/null"

MASTER_NAME="${INSTANCE_PREFIX}-master"
MINION_NAMES=($(eval echo ${INSTANCE_PREFIX}-minion-{1..${NUM_MINIONS}}))
MASTER_TAG="${INSTANCE_PREFIX}-master"
MINION_TAG="${INSTANCE_PREFIX}-minion"
MINION_IP_RANGES=($(eval echo "10.244.{1..${NUM_MINIONS}}.0/24"))
MINION_SCOPES=""
POLL_SLEEP_INTERVAL=3
PORTAL_NET="10.0.0.0/16"
MASTER_IP_RANGE="${MASTER_IP_RANGE:-10.246.0.0/24}"


# When set to true, Docker Cache is enabled by default as part of the cluster bring up.
ENABLE_DOCKER_REGISTRY_CACHE=true

# Optional: Install node monitoring.
ENABLE_NODE_MONITORING="${KUBE_ENABLE_NODE_MONITORING:-true}"

# Optional: When set to true, heapster, Influxdb and Grafana will be setup as part of the cluster bring up.
ENABLE_CLUSTER_MONITORING="${KUBE_ENABLE_CLUSTER_MONITORING:-true}"

# Optional: Enable node logging.
ENABLE_NODE_LOGGING="${KUBE_ENABLE_NODE_LOGGING:-true}"
LOGGING_DESTINATION="${KUBE_LOGGING_DESTINATION:-elasticsearch}" # options: elasticsearch, gcp

# Optional: When set to true, Elasticsearch and Kibana will be setup as part of the cluster bring up.
ENABLE_CLUSTER_LOGGING="${KUBE_ENABLE_CLUSTER_LOGGING:-true}"
ELASTICSEARCH_LOGGING_REPLICAS=1

# Don't require https for registries in our local RFC1918 network
EXTRA_DOCKER_OPTS="--insecure-registry 10.0.0.0/8"

# Optional: Install cluster DNS.
ENABLE_CLUSTER_DNS=true
DNS_SERVER_IP="10.0.0.10"
DNS_DOMAIN="kubernetes.local"
DNS_REPLICAS=1

# Admission Controllers to invoke prior to persisting objects in cluster
ADMISSION_CONTROL=NamespaceLifecycle,NamespaceAutoProvision,LimitRanger,SecurityContextDeny,ResourceQuota
