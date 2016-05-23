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

## Contains configuration values for interacting with the docker-in-docker cluster

NUM_NODES=${NUM_NODES:-2}
INSTANCE_PREFIX="${INSTANCE_PREFIX:-kubernetes}"
MASTER_NAME="${INSTANCE_PREFIX}-master"
NODE_NAMES=($(eval echo ${INSTANCE_PREFIX}-minion-{1..${NUM_NODES}}))

APISERVER_SERVICE_IP=${APISERVER_SERVICE_IP:-10.0.0.1}
SERVICE_CIDR="${SERVICE_CIDR:-${APISERVER_SERVICE_IP}/24}"

# Extra options to set on the Docker command line.  This is useful for setting
# --insecure-registry for local registries.
DOCKER_OPTS=""

# Optional: Deploy cluster DNS.
ENABLE_CLUSTER_DNS="${ENABLE_CLUSTER_DNS:-true}"
DNS_SERVER_IP="${DNS_SERVER_IP:-10.0.0.10}"
DNS_DOMAIN="${DNS_DOMAIN:-cluster.local}"
DNS_REPLICAS="${DNS_REPLICAS:-1}"

# Optional: Deploy cluster web interface.
ENABLE_CLUSTER_UI="${ENABLE_CLUSTER_UI:-true}"

# Timeout (in seconds) to wait for ssl certs to be generated
KUBE_KEYGEN_TIMEOUT="${KUBE_KEYGEN_TIMEOUT:-60}"

# Timeout (in seconds) to wait for Etcd to come up
DOCKER_IN_DOCKER_ETCD_TIMEOUT="${DOCKER_IN_DOCKER_ETCD_TIMEOUT:-60}"

# Timeout (in seconds) to wait for the API Server to come up
DOCKER_IN_DOCKER_API_TIMEOUT="${DOCKER_IN_DOCKER_API_TIMEOUT:-180}"

# Timeout (in seconds) to wait for each addon to come up
DOCKER_IN_DOCKER_ADDON_TIMEOUT="${DOCKER_IN_DOCKER_ADDON_TIMEOUT:-180}"

# Path to directory on the host to use as the root for multiple docker volumes.
# ${DOCKER_IN_DOCKER_WORK_DIR}/log - storage of component logs (written on deploy failure)
# ${DOCKER_IN_DOCKER_WORK_DIR}/auth - storage of SSL certs/keys/tokens
# If using docker-machine or boot2docker, should be under /Users (which is mounted from the host into the docker vm).
# If running in a container, $HOME should be resolved outside of the container.
DOCKER_IN_DOCKER_WORK_DIR="${DOCKER_IN_DOCKER_WORK_DIR:-${HOME}/tmp/kubernetes-dind}"

# Arguments to pass to docker-engine running on the kubernetes-dind containers.
DOCKER_DAEMON_ARGS="${DOCKER_DAEMON_ARGS:---log-level=error}"

# Force a dump of the logs into ${DOCKER_IN_DOCKER_WORK_DIR}/log on kube-down.sh. By
# default this only happens when kube-up.sh fails.
DOCKER_IN_DOCKER_DUMP_LOGS="${DOCKER_IN_DOCKER_DUMP_LOGS:-false}"

# Skip rebuilding the involved docker containers on kube-up.sh.
DOCKER_IN_DOCKER_SKIP_BUILD="${DOCKER_IN_DOCKER_SKIP_BUILD:-false}"
