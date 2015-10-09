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

# A library of helper functions and constant for debian os distro

# $1: if 'true', we're building a master yaml, else a node
function build-kube-env {
  local master=$1
  local file=$2

  rm -f ${file}
  cat >$file <<EOF
ENV_TIMESTAMP: $(yaml-quote $(date -u +%Y-%m-%dT%T%z))
INSTANCE_PREFIX: $(yaml-quote ${INSTANCE_PREFIX})
NODE_INSTANCE_PREFIX: $(yaml-quote ${NODE_INSTANCE_PREFIX})
CLUSTER_IP_RANGE: $(yaml-quote ${CLUSTER_IP_RANGE:-10.244.0.0/16})
SERVER_BINARY_TAR_URL: $(yaml-quote ${SERVER_BINARY_TAR_URL})
SERVER_BINARY_TAR_HASH: $(yaml-quote ${SERVER_BINARY_TAR_HASH})
SALT_TAR_URL: $(yaml-quote ${SALT_TAR_URL})
SALT_TAR_HASH: $(yaml-quote ${SALT_TAR_HASH})
SERVICE_CLUSTER_IP_RANGE: $(yaml-quote ${SERVICE_CLUSTER_IP_RANGE})
KUBERNETES_MASTER_NAME: $(yaml-quote ${MASTER_NAME})
ALLOCATE_NODE_CIDRS: $(yaml-quote ${ALLOCATE_NODE_CIDRS:-false})
ENABLE_CLUSTER_MONITORING: $(yaml-quote ${ENABLE_CLUSTER_MONITORING:-none})
ENABLE_CLUSTER_LOGGING: $(yaml-quote ${ENABLE_CLUSTER_LOGGING:-false})
ENABLE_CLUSTER_UI: $(yaml-quote ${ENABLE_CLUSTER_UI:-false})
ENABLE_NODE_LOGGING: $(yaml-quote ${ENABLE_NODE_LOGGING:-false})
LOGGING_DESTINATION: $(yaml-quote ${LOGGING_DESTINATION:-})
ELASTICSEARCH_LOGGING_REPLICAS: $(yaml-quote ${ELASTICSEARCH_LOGGING_REPLICAS:-})
ENABLE_CLUSTER_DNS: $(yaml-quote ${ENABLE_CLUSTER_DNS:-false})
ENABLE_CLUSTER_REGISTRY: $(yaml-quote ${ENABLE_CLUSTER_REGISTRY:-false})
CLUSTER_REGISTRY_DISK: $(yaml-quote ${CLUSTER_REGISTRY_DISK})
CLUSTER_REGISTRY_DISK_SIZE: $(yaml-quote ${CLUSTER_REGISTRY_DISK_SIZE})
DNS_REPLICAS: $(yaml-quote ${DNS_REPLICAS:-})
DNS_SERVER_IP: $(yaml-quote ${DNS_SERVER_IP:-})
DNS_DOMAIN: $(yaml-quote ${DNS_DOMAIN:-})
KUBELET_TOKEN: $(yaml-quote ${KUBELET_TOKEN:-})
KUBE_PROXY_TOKEN: $(yaml-quote ${KUBE_PROXY_TOKEN:-})
ADMISSION_CONTROL: $(yaml-quote ${ADMISSION_CONTROL:-})
MASTER_IP_RANGE: $(yaml-quote ${MASTER_IP_RANGE})
ENABLE_HORIZONTAL_POD_AUTOSCALER: $(yaml-quote ${ENABLE_HORIZONTAL_POD_AUTOSCALER})
ENABLE_DEPLOYMENTS: $(yaml-quote ${ENABLE_DEPLOYMENTS})
RUNTIME_CONFIG: $(yaml-quote ${RUNTIME_CONFIG})
CA_CERT: $(yaml-quote ${CA_CERT_BASE64:-})
KUBELET_CERT: $(yaml-quote ${KUBELET_CERT_BASE64:-})
KUBELET_KEY: $(yaml-quote ${KUBELET_KEY_BASE64:-})
EOF
  if [ -n "${KUBE_APISERVER_REQUEST_TIMEOUT:-}" ]; then
    cat >>$file <<EOF
KUBE_APISERVER_REQUEST_TIMEOUT: $(yaml-quote ${KUBE_APISERVER_REQUEST_TIMEOUT})
EOF
  fi
  if [ -n "${TERMINATED_POD_GC_THRESHOLD:-}" ]; then
    cat >>$file <<EOF
TERMINATED_POD_GC_THRESHOLD: $(yaml-quote ${TERMINATED_POD_GC_THRESHOLD})
EOF
  fi
  if [ -n "${TEST_CLUSTER:-}" ]; then
    cat >>$file <<EOF
TEST_CLUSTER: $(yaml-quote ${TEST_CLUSTER})
EOF
  fi
  if [[ "${master}" == "true" ]]; then
    # Master-only env vars.
    cat >>$file <<EOF
KUBERNETES_MASTER: "true"
KUBE_USER: $(yaml-quote ${KUBE_USER})
KUBE_PASSWORD: $(yaml-quote ${KUBE_PASSWORD})
KUBE_BEARER_TOKEN: $(yaml-quote ${KUBE_BEARER_TOKEN})
MASTER_CERT: $(yaml-quote ${MASTER_CERT_BASE64:-})
MASTER_KEY: $(yaml-quote ${MASTER_KEY_BASE64:-})
KUBECFG_CERT: $(yaml-quote ${KUBECFG_CERT_BASE64:-})
KUBECFG_KEY: $(yaml-quote ${KUBECFG_KEY_BASE64:-})
KUBELET_APISERVER: $(yaml-quote ${KUBELET_APISERVER:-})
ENABLE_MANIFEST_URL: $(yaml-quote ${ENABLE_MANIFEST_URL:-false})
MANIFEST_URL: $(yaml-quote ${MANIFEST_URL:-})
MANIFEST_URL_HEADER: $(yaml-quote ${MANIFEST_URL_HEADER:-})
NUM_MINIONS: $(yaml-quote ${NUM_MINIONS})
EOF
    if [ -n "${APISERVER_TEST_ARGS:-}" ]; then
      cat >>$file <<EOF
APISERVER_TEST_ARGS: $(yaml-quote ${APISERVER_TEST_ARGS})
EOF
    fi
    if [ -n "${KUBELET_TEST_ARGS:-}" ]; then
      cat >>$file <<EOF
KUBELET_TEST_ARGS: $(yaml-quote ${KUBELET_TEST_ARGS})
EOF
    fi
    if [ -n "${CONTROLLER_MANAGER_TEST_ARGS:-}" ]; then
      cat >>$file <<EOF
CONTROLLER_MANAGER_TEST_ARGS: $(yaml-quote ${CONTROLLER_MANAGER_TEST_ARGS})
EOF
    fi
    if [ -n "${SCHEDULER_TEST_ARGS:-}" ]; then
      cat >>$file <<EOF
SCHEDULER_TEST_ARGS: $(yaml-quote ${SCHEDULER_TEST_ARGS})
EOF
    fi
  else
    # Node-only env vars.
    cat >>$file <<EOF
KUBERNETES_MASTER: "false"
ZONE: $(yaml-quote ${ZONE})
EXTRA_DOCKER_OPTS: $(yaml-quote ${EXTRA_DOCKER_OPTS:-})
EOF
    if [ -n "${KUBELET_TEST_ARGS:-}" ]; then
      cat >>$file <<EOF
KUBELET_TEST_ARGS: $(yaml-quote ${KUBELET_TEST_ARGS})
EOF
    fi
    if [ -n "${KUBEPROXY_TEST_ARGS:-}" ]; then
      cat >>$file <<EOF
KUBEPROXY_TEST_ARGS: $(yaml-quote ${KUBEPROXY_TEST_ARGS})
EOF
    fi
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
    --scopes "storage-ro,compute-rw,monitoring,logging-write" \
    --can-ip-forward \
    --metadata-from-file \
      "startup-script=${KUBE_ROOT}/cluster/gce/configure-vm.sh,kube-env=${KUBE_TEMP}/master-kube-env.yaml" \
    --disk "name=${MASTER_NAME}-pd,device-name=master-pd,mode=rw,boot=no,auto-delete=no"
}

# TODO(zmerlynn): Make $1 required.
# TODO(zmerlynn): Document required vars (for this and call chain).
# $1 version
function create-node-instance-template {
  local suffix=""
  if [[ -n ${1:-} ]]; then
    suffix="-${1}"
  fi
  create-node-template "${NODE_INSTANCE_PREFIX}-template${suffix}" "${scope_flags}" \
    "startup-script=${KUBE_ROOT}/cluster/gce/configure-vm.sh" \
    "kube-env=${KUBE_TEMP}/node-kube-env.yaml"
}
