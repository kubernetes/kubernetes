#!/bin/bash

# Copyright 2014 Google Inc. All rights reserved.
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

# TODO(jbeda): Provide a way to override project
# gcloud multiplexing for shared GCE/GKE tests.
GCLOUD=gcloud
ZONE=${KUBE_GCE_ZONE:-us-central1-b}
MASTER_SIZE=g1-small
MINION_SIZE=g1-small
NUM_MINIONS=${NUM_MINIONS:-2}
MINION_DISK_TYPE=pd-standard
MINION_DISK_SIZE=10GB
# TODO(dchen1107): Filed an internal issue to create an alias
# for containervm image, so that gcloud will expand this
# to the latest supported image.
IMAGE=container-vm-v20150112
IMAGE_PROJECT=google-containers
NETWORK=${KUBE_GCE_NETWORK:-e2e}
INSTANCE_PREFIX="${KUBE_GCE_INSTANCE_PREFIX:-e2e-test-${USER}}"
MASTER_NAME="${INSTANCE_PREFIX}-master"
MASTER_TAG="${INSTANCE_PREFIX}-master"
MINION_TAG="${INSTANCE_PREFIX}-minion"
MINION_NAMES=($(eval echo ${INSTANCE_PREFIX}-minion-{1..${NUM_MINIONS}}))
CLUSTER_IP_RANGE="10.245.0.0/16"
MINION_IP_RANGES=($(eval echo "10.245.{1..${NUM_MINIONS}}.0/24"))
MINION_SCOPES=("storage-ro" "compute-rw")
# Increase the sleep interval value if concerned about API rate limits. 3, in seconds, is the default.
POLL_SLEEP_INTERVAL=3
PORTAL_NET="10.0.0.0/16"

# When set to true, Docker Cache is enabled by default as part of the cluster bring up.
ENABLE_DOCKER_REGISTRY_CACHE=true

ENABLE_NODE_MONITORING=true

ENABLE_NODE_LOGGING=true
LOGGING_DESTINATION=elasticsearch # options: elasticsearch, gcp

ENABLE_CLUSTER_MONITORING=false

# Don't require https for registries in our local RFC1918 network
EXTRA_DOCKER_OPTS="--insecure-registry 10.0.0.0/8"

# Optional: Install cluster DNS.
ENABLE_CLUSTER_DNS=true
DNS_SERVER_IP="10.0.0.10"
DNS_DOMAIN="kubernetes.local"
DNS_REPLICAS=1
