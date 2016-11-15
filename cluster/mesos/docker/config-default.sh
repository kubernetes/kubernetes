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

## Contains configuration values for interacting with the mesos/docker cluster

NUM_NODES=${NUM_NODES:-2}
INSTANCE_PREFIX="${INSTANCE_PREFIX:-kubernetes}"
MASTER_NAME="${INSTANCE_PREFIX}-master"
NODE_NAMES=($(eval echo ${INSTANCE_PREFIX}-minion-{1..${NUM_NODES}}))

SERVICE_CLUSTER_IP_RANGE=10.10.10.0/24

# Extra options to set on the Docker command line.  This is useful for setting
# --insecure-registry for local registries.
DOCKER_OPTS=""

# Optional: Deploy cluster DNS.
#ENABLE_CLUSTER_DNS=false
ENABLE_CLUSTER_DNS=true
DNS_SERVER_IP="10.10.10.10"
DNS_DOMAIN="cluster.local"

# Optional: Enable DNS horizontal autoscaler
ENABLE_DNS_HORIZONTAL_AUTOSCALER="${KUBE_ENABLE_DNS_HORIZONTAL_AUTOSCALER:-false}"

# Optional: Deploy cluster web interface.
ENABLE_CLUSTER_UI=true

# Timeout (in seconds) to wait for ssl certs to be generated
KUBE_KEYGEN_TIMEOUT="${KUBE_KEYGEN_TIMEOUT:-60}"

# Timeout (in seconds) to wait for Etcd to come up
MESOS_DOCKER_ETCD_TIMEOUT="${MESOS_DOCKER_ETCD_TIMEOUT:-60}"

# Timeout (in seconds) to wait for the Mesos Master to come up
MESOS_DOCKER_MESOS_TIMEOUT="${MESOS_DOCKER_MESOS_TIMEOUT:-60}"

# Timeout (in seconds) to wait for the API Server to come up
MESOS_DOCKER_API_TIMEOUT="${MESOS_DOCKER_API_TIMEOUT:-180}"

# Timeout (in seconds) to wait for each addon to come up
MESOS_DOCKER_ADDON_TIMEOUT="${MESOS_DOCKER_ADDON_TIMEOUT:-180}"

# Path to directory on the host to use as the root for multiple docker volumes.
# ${MESOS_DOCKER_WORK_DIR}/log - storage of component logs (written on deploy failure)
# ${MESOS_DOCKER_WORK_DIR}/auth - storage of SSL certs/keys/tokens
# ${MESOS_DOCKER_WORK_DIR}/<component>/mesos - storage of mesos slave work (e.g. task logs)
# If using docker-machine or Docker for Mac, should be under /Users (which is mounted from the host into the docker vm).
# If running in a container, $HOME should be resolved outside of the container.
MESOS_DOCKER_WORK_DIR="${MESOS_DOCKER_WORK_DIR:-${HOME}/tmp/kubernetes}"

# Arguments to pass to docker-engine running on the mesos-slave-dind containers.
DOCKER_DAEMON_ARGS="${DOCKER_DAEMON_ARGS:---log-level=error}"

# Force a dump of the logs into ${MESOS_DOCKER_WORK_DIR}/log on kube-down.sh. By
# default this only happens when kube-up.sh fails.
MESOS_DOCKER_DUMP_LOGS="${MESOS_DOCKER_DUMP_LOGS:-false}"

# Skip rebuilding the involved docker containers on kube-up.sh.
MESOS_DOCKER_SKIP_BUILD="${MESOS_DOCKER_SKIP_BUILD:-false}"
