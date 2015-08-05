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

## Contains configuration values for interacting with the mesos/docker cluster

NUM_MINIONS=${NUM_MINIONS:-2}
INSTANCE_PREFIX="${INSTANCE_PREFIX:-kubernetes}"
MASTER_NAME="${INSTANCE_PREFIX}-master"
MINION_NAMES=($(eval echo ${INSTANCE_PREFIX}-minion-{1..${NUM_MINIONS}}))

SERVICE_CLUSTER_IP_RANGE=10.10.10.0/24

# Extra options to set on the Docker command line.  This is useful for setting
# --insecure-registry for local registries.
DOCKER_OPTS=""

# Optional: Deploy cluster DNS.
#ENABLE_CLUSTER_DNS=false
ENABLE_CLUSTER_DNS=true
DNS_SERVER_IP="10.10.10.10"
DNS_DOMAIN="cluster.local"
DNS_REPLICAS=1

# Optional: Deploy cluster web interface.
ENABLE_CLUSTER_UI=true
