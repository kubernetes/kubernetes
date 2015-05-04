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

# A library of helper functions and constant for coreos os distro

# $1: if 'true', we're building a master yaml, else a node
function build-kube-env {
  local master=$1
  local file=$2

  rm -f ${file}
  # TODO(dawnchen): master node is still running with debian image
  if [[ "${master}" == "true" ]]; then
  cat >$file <<EOF
ENV_TIMESTAMP: $(yaml-quote $(date -u +%Y-%m-%dT%T%z))
INSTANCE_PREFIX: $(yaml-quote ${INSTANCE_PREFIX})
NODE_INSTANCE_PREFIX: $(yaml-quote ${NODE_INSTANCE_PREFIX})
KUBE_GCE_CLUSTER_CLASS_B: $(yaml-quote ${KUBE_GCE_CLUSTER_CLASS_B:-10.244})
SERVER_BINARY_TAR_URL: $(yaml-quote ${SERVER_BINARY_TAR_URL})
SALT_TAR_URL: $(yaml-quote ${SALT_TAR_URL})
PORTAL_NET: $(yaml-quote ${PORTAL_NET})
ALLOCATE_NODE_CIDRS: $(yaml-quote ${ALLOCATE_NODE_CIDRS:-false})
ENABLE_CLUSTER_MONITORING: $(yaml-quote ${ENABLE_CLUSTER_MONITORING:-none})
ENABLE_NODE_MONITORING: $(yaml-quote ${ENABLE_NODE_MONITORING:-false})
ENABLE_CLUSTER_LOGGING: $(yaml-quote ${ENABLE_CLUSTER_LOGGING:-false})
ENABLE_NODE_LOGGING: $(yaml-quote ${ENABLE_NODE_LOGGING:-false})
LOGGING_DESTINATION: $(yaml-quote ${LOGGING_DESTINATION:-})
ELASTICSEARCH_LOGGING_REPLICAS: $(yaml-quote ${ELASTICSEARCH_LOGGING_REPLICAS:-})
ENABLE_CLUSTER_DNS: $(yaml-quote ${ENABLE_CLUSTER_DNS:-false})
DNS_REPLICAS: $(yaml-quote ${DNS_REPLICAS:-})
DNS_SERVER_IP: $(yaml-quote ${DNS_SERVER_IP:-})
DNS_DOMAIN: $(yaml-quote ${DNS_DOMAIN:-})
KUBE_USER: $(yaml-quote ${KUBE_USER})
KUBE_PASSWORD: $(yaml-quote ${KUBE_PASSWORD})
KUBE_BEARER_TOKEN: $(yaml-quote ${KUBE_BEARER_TOKEN})
KUBELET_TOKEN: $(yaml-quote ${KUBELET_TOKEN:-})
KUBE_PROXY_TOKEN: $(yaml-quote ${KUBE_PROXY_TOKEN:-})
ADMISSION_CONTROL: $(yaml-quote ${ADMISSION_CONTROL:-})
MASTER_IP_RANGE: $(yaml-quote ${MASTER_IP_RANGE})
KUBERNETES_CONTAINER_RUNTIME: $(yaml-quote ${CONTAINER_RUNTIME})
EOF
  else
    cat >>$file <<EOF
ENV_TIMESTAMP=$(yaml-quote $(date -u +%Y-%m-%dT%T%z))
INSTANCE_PREFIX=$(yaml-quote ${INSTANCE_PREFIX})
NODE_INSTANCE_PREFIX=$(yaml-quote ${NODE_INSTANCE_PREFIX})
SERVER_BINARY_TAR_URL=$(yaml-quote ${SERVER_BINARY_TAR_URL})
PORTAL_NET=$(yaml-quote ${PORTAL_NET})
ENABLE_CLUSTER_MONITORING=$(yaml-quote ${ENABLE_CLUSTER_MONITORING:-none})
ENABLE_NODE_MONITORING=$(yaml-quote ${ENABLE_NODE_MONITORING:-false})
ENABLE_CLUSTER_LOGGING=$(yaml-quote ${ENABLE_CLUSTER_LOGGING:-false})
ENABLE_NODE_LOGGING=$(yaml-quote ${ENABLE_NODE_LOGGING:-false})
LOGGING_DESTINATION=$(yaml-quote ${LOGGING_DESTINATION:-})
ELASTICSEARCH_LOGGING_REPLICAS=$(yaml-quote ${ELASTICSEARCH_LOGGING_REPLICAS:-})
ENABLE_CLUSTER_DNS=$(yaml-quote ${ENABLE_CLUSTER_DNS:-false})
DNS_REPLICAS=$(yaml-quote ${DNS_REPLICAS:-})
DNS_SERVER_IP=$(yaml-quote ${DNS_SERVER_IP:-})
DNS_DOMAIN=$(yaml-quote ${DNS_DOMAIN:-})
KUBE_USER=$(yaml-quote ${KUBE_USER})
KUBE_PASSWORD=$(yaml-quote ${KUBE_PASSWORD})
KUBE_BEARER_TOKEN=$(yaml-quote ${KUBE_BEARER_TOKEN})
KUBELET_TOKEN=$(yaml-quote ${KUBELET_TOKEN:-})
KUBE_PROXY_TOKEN=$(yaml-quote ${KUBE_PROXY_TOKEN:-})
ADMISSION_CONTROL=$(yaml-quote ${ADMISSION_CONTROL:-})
MASTER_IP_RANGE=$(yaml-quote ${MASTER_IP_RANGE})
KUBERNETES_MASTER_NAME=$(yaml-quote ${MASTER_NAME})
ZONE=$(yaml-quote ${ZONE})
EXTRA_DOCKER_OPTS=$(yaml-quote ${EXTRA_DOCKER_OPTS})
ENABLE_DOCKER_REGISTRY_CACHE=$(yaml-quote ${ENABLE_DOCKER_REGISTRY_CACHE:-false})
PROJECT_ID=$(yaml-quote ${PROJECT})
KUBERNETES_CONTAINER_RUNTIME=$(yaml-quote ${CONTAINER_RUNTIME})
EOF
  fi
}

# create-master-instance creates the master instance. If called with
# an argument, the argument is used as the name to a reserved IP
# address for the master. (In the case of upgrade/repair, we re-use
# the same IP.)
#
# It requires a whole slew of assumed variables, partially due to to
# the call to write-master-env. Listing them would be rather
# futile. Instead, we list the required calls to ensure any additional
# variables are set:
#   ensure-temp-dir
#   detect-project
#   get-bearer-token
#
# TODO(dawnchen): Convert master node to use coreos image too
function create-master-instance {
  local address_opt=""
  [[ -n ${1:-} ]] && address_opt="--address ${1}"

  write-master-env
  gcloud compute instances create "${MASTER_NAME}" \
    ${address_opt} \
    --project "${PROJECT}" \
    --zone "${ZONE}" \
    --machine-type "${MASTER_SIZE}" \
    --image-project="${MASTER_IMAGE_PROJECT}" \
    --image "${MASTER_IMAGE}" \
    --tags "${MASTER_TAG}" \
    --network "${NETWORK}" \
    --scopes "storage-ro" "compute-rw" \
    --can-ip-forward \
    --metadata-from-file \
      "startup-script=${KUBE_ROOT}/cluster/gce/configure-vm.sh" \
      "kube-env=${KUBE_TEMP}/master-kube-env.yaml" \
    --disk name="${MASTER_NAME}-pd" device-name=master-pd mode=rw boot=no auto-delete=no
}

# TODO(dawnchen): Check $CONTAINER_RUNTIME to decide which
# cloud_config yaml file should be passed
function create-node-instance-template {
   create-node-template "${NODE_INSTANCE_PREFIX}-template" "${scope_flags[*]}" \
    "kube-env=${KUBE_TEMP}/node-kube-env.yaml" \
    "user-data=${KUBE_ROOT}/cluster/gce/coreos/node.yaml"
}
