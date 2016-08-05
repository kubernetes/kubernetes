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

ZONE=${KUBE_AWS_ZONE:-us-west-2a}
MASTER_SIZE=${MASTER_SIZE:-}
NODE_SIZE=${NODE_SIZE:-}
NUM_NODES=${NUM_NODES:-4}

# Dynamically set node sizes so that Heapster has enough space to run
if [[ -z ${NODE_SIZE} ]]; then
  if (( ${NUM_NODES} < 50 )); then
    NODE_SIZE="t2.micro"
  elif (( ${NUM_NODES} < 150 )); then
    NODE_SIZE="t2.small"
  else
    NODE_SIZE="t2.medium"
  fi
fi

# Dynamically set the master size by the number of nodes, these are guesses
if [[ -z ${MASTER_SIZE} ]]; then
  MASTER_SIZE="m3.medium"
  if [[ "${NUM_NODES}" -gt "5" ]]; then
    suggested_master_size="m3.large"
  fi
  if [[ "${NUM_NODES}" -gt "10" ]]; then
    suggested_master_size="m3.xlarge"
  fi
  if [[ "${NUM_NODES}" -gt "100" ]]; then
    suggested_master_size="m3.2xlarge"
  fi
  if [[ "${NUM_NODES}" -gt "250" ]]; then
    suggested_master_size="c4.4xlarge"
  fi
  if [[ "${NUM_NODES}" -gt "500" ]]; then
    suggested_master_size="c4.8xlarge"
  fi
fi

# Optional: Set AWS_S3_BUCKET to the name of an S3 bucket to use for uploading binaries
# (otherwise a unique bucket name will be generated for you)
#  AWS_S3_BUCKET=kubernetes-artifacts

# Because regions are globally named, we want to create in a single region; default to us-east-1
AWS_S3_REGION=${AWS_S3_REGION:-us-east-1}

# Which docker storage mechanism to use.
DOCKER_STORAGE=${DOCKER_STORAGE:-aufs}

# Extra docker options for nodes.
EXTRA_DOCKER_OPTS="${EXTRA_DOCKER_OPTS:-}"

INSTANCE_PREFIX="${KUBE_AWS_INSTANCE_PREFIX:-kubernetes}"
CLUSTER_ID=${INSTANCE_PREFIX}
AWS_SSH_KEY=${AWS_SSH_KEY:-$HOME/.ssh/kube_aws_rsa}
CONFIG_CONTEXT="${KUBE_CONFIG_CONTEXT:-aws_${INSTANCE_PREFIX}}"
IAM_PROFILE_MASTER="kubernetes-master"
IAM_PROFILE_NODE="kubernetes-minion"

LOG="/dev/null"

MASTER_DISK_TYPE="${MASTER_DISK_TYPE:-gp2}"
MASTER_DISK_SIZE=${MASTER_DISK_SIZE:-20}
# The master root EBS volume size (typically does not need to be very large)
MASTER_ROOT_DISK_TYPE="${MASTER_ROOT_DISK_TYPE:-gp2}"
MASTER_ROOT_DISK_SIZE=${MASTER_ROOT_DISK_SIZE:-8}
# The minions root EBS volume size (used to house Docker images)
NODE_ROOT_DISK_TYPE="${NODE_ROOT_DISK_TYPE:-gp2}"
NODE_ROOT_DISK_SIZE=${NODE_ROOT_DISK_SIZE:-32}

MASTER_NAME="${INSTANCE_PREFIX}-master"
MASTER_TAG="${INSTANCE_PREFIX}-master"
NODE_TAG="${INSTANCE_PREFIX}-minion"
NODE_SCOPES=""
NON_MASQUERADE_CIDR="${NON_MASQUERADE_CIDR:-10.0.0.0/8}" # Traffic to IPs outside this range will use IP masquerade
SERVICE_CLUSTER_IP_RANGE="${SERVICE_CLUSTER_IP_RANGE:-10.0.0.0/16}"  # formerly PORTAL_NET
CLUSTER_IP_RANGE="${CLUSTER_IP_RANGE:-10.244.0.0/16}"
MASTER_IP_RANGE="${MASTER_IP_RANGE:-10.246.0.0/24}"
# If set to an Elastic IP address, the master instance will be associated with this IP.
# Otherwise a new Elastic IP will be acquired
# (We used to accept 'auto' to mean 'allocate elastic ip', but that is now the default)
MASTER_RESERVED_IP="${MASTER_RESERVED_IP:-}"

# Runtime config
RUNTIME_CONFIG="${KUBE_RUNTIME_CONFIG:-}"

# Optional: Cluster monitoring to setup as part of the cluster bring up:
#   none     - No cluster monitoring setup
#   influxdb - Heapster, InfluxDB, and Grafana
ENABLE_CLUSTER_MONITORING="${KUBE_ENABLE_CLUSTER_MONITORING:-influxdb}"

# Optional: Enable node logging.
ENABLE_NODE_LOGGING="${KUBE_ENABLE_NODE_LOGGING:-true}"
LOGGING_DESTINATION="${KUBE_LOGGING_DESTINATION:-elasticsearch}" # options: elasticsearch, gcp

# Optional: When set to true, Elasticsearch and Kibana will be setup as part of the cluster bring up.
ENABLE_CLUSTER_LOGGING="${KUBE_ENABLE_CLUSTER_LOGGING:-true}"
ELASTICSEARCH_LOGGING_REPLICAS=1

# Optional: Don't require https for registries in our local RFC1918 network
if [[ ${KUBE_ENABLE_INSECURE_REGISTRY:-false} == "true" ]]; then
  EXTRA_DOCKER_OPTS="${EXTRA_DOCKER_OPTS} --insecure-registry ${NON_MASQUERADE_CIDR}"
fi

# Optional: Install cluster DNS.
ENABLE_CLUSTER_DNS="${KUBE_ENABLE_CLUSTER_DNS:-true}"
DNS_SERVER_IP="${DNS_SERVER_IP:-10.0.0.10}"
DNS_DOMAIN="cluster.local"
DNS_REPLICAS=1

# Optional: Install Kubernetes UI
ENABLE_CLUSTER_UI="${KUBE_ENABLE_CLUSTER_UI:-true}"

# Optional: Create autoscaler for cluster's nodes.
ENABLE_CLUSTER_AUTOSCALER="${KUBE_ENABLE_CLUSTER_AUTOSCALER:-false}"
if [[ "${ENABLE_CLUSTER_AUTOSCALER}" == "true" ]]; then
  # TODO: actually configure ASG or similar
  AUTOSCALER_MIN_NODES="${KUBE_AUTOSCALER_MIN_NODES:-1}"
  AUTOSCALER_MAX_NODES="${KUBE_AUTOSCALER_MAX_NODES:-${NUM_NODES}}"
  TARGET_NODE_UTILIZATION="${KUBE_TARGET_NODE_UTILIZATION:-0.7}"
fi

# Admission Controllers to invoke prior to persisting objects in cluster
# If we included ResourceQuota, we should keep it at the end of the list to prevent incremeting quota usage prematurely.
ADMISSION_CONTROL=NamespaceLifecycle,LimitRanger,ServiceAccount,PersistentVolumeLabel,ResourceQuota

# Optional: Enable/disable public IP assignment for minions.
# Important Note: disable only if you have setup a NAT instance for internet access and configured appropriate routes!
ENABLE_NODE_PUBLIC_IP=${KUBE_ENABLE_NODE_PUBLIC_IP:-true}

# OS options for minions
KUBE_OS_DISTRIBUTION="${KUBE_OS_DISTRIBUTION:-jessie}"
MASTER_OS_DISTRIBUTION="${KUBE_OS_DISTRIBUTION}"
NODE_OS_DISTRIBUTION="${KUBE_OS_DISTRIBUTION}"
KUBE_NODE_IMAGE="${KUBE_NODE_IMAGE:-}"
COREOS_CHANNEL="${COREOS_CHANNEL:-alpha}"
CONTAINER_RUNTIME="${KUBE_CONTAINER_RUNTIME:-docker}"
RKT_VERSION="${KUBE_RKT_VERSION:-0.5.5}"

# OpenContrail networking plugin specific settings
NETWORK_PROVIDER="${NETWORK_PROVIDER:-none}" # opencontrail
OPENCONTRAIL_TAG="${OPENCONTRAIL_TAG:-R2.20}"
OPENCONTRAIL_KUBERNETES_TAG="${OPENCONTRAIL_KUBERNETES_TAG:-master}"
OPENCONTRAIL_PUBLIC_SUBNET="${OPENCONTRAIL_PUBLIC_SUBNET:-10.1.0.0/16}"

# Optional: if set to true, kube-up will configure the cluster to run e2e tests.
E2E_STORAGE_TEST_ENVIRONMENT=${KUBE_E2E_STORAGE_TEST_ENVIRONMENT:-false}
