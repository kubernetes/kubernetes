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

# Use the gcloud defaults to find the project.  If it is already set in the
# environment then go with that.
#
# Vars set:
#   PROJECT
#   PROJECT_REPORTED
function detect-project () {
  if [[ -z "${PROJECT-}" ]]; then
    PROJECT=$(gcloud config list project | tail -n 1 | cut -f 3 -d ' ')
  fi

  if [[ -z "${PROJECT-}" ]]; then
    echo "Could not detect Google Cloud Platform project.  Set the default project using " >&2
    echo "'gcloud config set project <PROJECT>'" >&2
    exit 1
  fi
  if [[ -z "${PROJECT_REPORTED-}" ]]; then
    echo "Project: ${PROJECT}" >&2
    echo "Zone: ${ZONE}" >&2
    PROJECT_REPORTED=true
  fi
}

# Wait for background jobs to finish. Exit with
# an error status if any of the jobs failed.
function wait-for-jobs {
  local fail=0
  local job
  for job in $(jobs -p); do
    wait "${job}" || fail=$((fail + 1))
  done
  if (( fail != 0 )); then
    echo -e "${color_red}${fail} commands failed.  Exiting.${color_norm}" >&2
    # Ignore failures for now.
    # exit 2
  fi
}

# Robustly try to create a firewall rule.
# $1: The name of firewall rule.
# $2: IP ranges.
# $3: Target tags for this firewall rule.
function create-firewall-rule {
  detect-project

  local attempt=0
  while true; do
    if ! gcloud compute firewall-rules create "$1" \
      --project "${PROJECT}" \
      --network "${NETWORK}" \
      --source-ranges "$2" \
      --target-tags "$3" \
      --allow tcp,udp,icmp,esp,ah,sctp; then
      if (( attempt > 4 )); then
        echo -e "${color_red}Failed to create firewall rule $1 ${color_norm}" >&2
        exit 2
      fi
      echo -e "${color_yellow}Attempt $(($attempt+1)) failed to create firewall rule $1. Retrying.${color_norm}" >&2
      attempt=$(($attempt+1))
      sleep $(($attempt * 5))
    else
      break
    fi
  done
}

# Quote something appropriate for a yaml string.
#
# TODO(zmerlynn): Note that this function doesn't so much "quote" as
# "strip out quotes", and we really should be using a YAML library for
# this, but PyYAML isn't shipped by default, and *rant rant rant ... SIGH*
function yaml-quote {
  echo "'$(echo "${@}" | sed -e "s/'/''/g")'"
}

function write-node-env {
  build-kube-env false "${KUBE_TEMP}/node-kube-env.yaml"
}

# TODO: Move this to master.sh once all the master related code is there.
function write-master-env {
  # If the user requested that the master be part of the cluster, set the
  # environment variable to program the master kubelet to register itself.
  if [[ "${REGISTER_MASTER_KUBELET:-}" == "true" ]]; then
    KUBELET_APISERVER="${MASTER_NAME}"
  fi

  build-kube-env true "${KUBE_TEMP}/master-kube-env.yaml"
}

# $1: if 'true', we're building a master yaml, else a node
function build-kube-env {
  local master=$1
  local file=$2

  build-runtime-config

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
ENABLE_L7_LOADBALANCING: $(yaml-quote ${ENABLE_L7_LOADBALANCING:-none})
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
RUNTIME_CONFIG: $(yaml-quote ${RUNTIME_CONFIG})
CA_CERT: $(yaml-quote ${CA_CERT_BASE64:-})
KUBELET_CERT: $(yaml-quote ${KUBELET_CERT_BASE64:-})
KUBELET_KEY: $(yaml-quote ${KUBELET_KEY_BASE64:-})
NETWORK_PROVIDER: $(yaml-quote ${NETWORK_PROVIDER:-})
OPENCONTRAIL_TAG: $(yaml-quote ${OPENCONTRAIL_TAG:-})
OPENCONTRAIL_KUBERNETES_TAG: $(yaml-quote ${OPENCONTRAIL_KUBERNETES_TAG:-})
OPENCONTRAIL_PUBLIC_SUBNET: $(yaml-quote ${OPENCONTRAIL_PUBLIC_SUBNET:-})
E2E_STORAGE_TEST_ENVIRONMENT: $(yaml-quote ${E2E_STORAGE_TEST_ENVIRONMENT:-})
KUBE_IMAGE_TAG: $(yaml-quote ${KUBE_IMAGE_TAG:-})
KUBE_DOCKER_REGISTRY: $(yaml-quote ${KUBE_DOCKER_REGISTRY:-})
EOF
  if [ -n "${KUBELET_PORT:-}" ]; then
    cat >>$file <<EOF
KUBELET_PORT: $(yaml-quote ${KUBELET_PORT})
EOF
  fi
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
  if [[ "${OS_DISTRIBUTION}" == "trusty" ]]; then
    cat >>$file <<EOF
KUBE_MANIFESTS_TAR_URL: $(yaml-quote ${KUBE_MANIFESTS_TAR_URL})
KUBE_MANIFESTS_TAR_HASH: $(yaml-quote ${KUBE_MANIFESTS_TAR_HASH})
EOF
  fi
  if [ -n "${TEST_CLUSTER:-}" ]; then
    cat >>$file <<EOF
TEST_CLUSTER: $(yaml-quote ${TEST_CLUSTER})
EOF
  fi
  if [ -n "${KUBELET_TEST_ARGS:-}" ]; then
      cat >>$file <<EOF
KUBELET_TEST_ARGS: $(yaml-quote ${KUBELET_TEST_ARGS})
EOF
  fi
  if [ -n "${KUBELET_TEST_LOG_LEVEL:-}" ]; then
      cat >>$file <<EOF
KUBELET_TEST_LOG_LEVEL: $(yaml-quote ${KUBELET_TEST_LOG_LEVEL})
EOF
  fi
  if [[ "${master}" == "true" ]]; then
    # Master-only env vars.
    cat >>$file <<EOF
KUBERNETES_MASTER: $(yaml-quote "true")
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
NUM_NODES: $(yaml-quote ${NUM_NODES})
EOF
    if [ -n "${APISERVER_TEST_ARGS:-}" ]; then
      cat >>$file <<EOF
APISERVER_TEST_ARGS: $(yaml-quote ${APISERVER_TEST_ARGS})
EOF
    fi
    if [ -n "${APISERVER_TEST_LOG_LEVEL:-}" ]; then
      cat >>$file <<EOF
APISERVER_TEST_LOG_LEVEL: $(yaml-quote ${APISERVER_TEST_LOG_LEVEL})
EOF
    fi
    if [ -n "${CONTROLLER_MANAGER_TEST_ARGS:-}" ]; then
      cat >>$file <<EOF
CONTROLLER_MANAGER_TEST_ARGS: $(yaml-quote ${CONTROLLER_MANAGER_TEST_ARGS})
EOF
    fi
    if [ -n "${CONTROLLER_MANAGER_TEST_LOG_LEVEL:-}" ]; then
      cat >>$file <<EOF
CONTROLLER_MANAGER_TEST_LOG_LEVEL: $(yaml-quote ${CONTROLLER_MANAGER_TEST_LOG_LEVEL})
EOF
    fi
    if [ -n "${SCHEDULER_TEST_ARGS:-}" ]; then
      cat >>$file <<EOF
SCHEDULER_TEST_ARGS: $(yaml-quote ${SCHEDULER_TEST_ARGS})
EOF
    fi
    if [ -n "${SCHEDULER_TEST_LOG_LEVEL:-}" ]; then
      cat >>$file <<EOF
SCHEDULER_TEST_LOG_LEVEL: $(yaml-quote ${SCHEDULER_TEST_LOG_LEVEL})
EOF
    fi
  else
    # Node-only env vars.
    cat >>$file <<EOF
KUBERNETES_MASTER: $(yaml-quote "false")
ZONE: $(yaml-quote ${ZONE})
EXTRA_DOCKER_OPTS: $(yaml-quote ${EXTRA_DOCKER_OPTS:-})
EOF
    if [ -n "${KUBEPROXY_TEST_ARGS:-}" ]; then
      cat >>$file <<EOF
KUBEPROXY_TEST_ARGS: $(yaml-quote ${KUBEPROXY_TEST_ARGS})
EOF
    fi
    if [ -n "${KUBEPROXY_TEST_LOG_LEVEL:-}" ]; then
      cat >>$file <<EOF
KUBEPROXY_TEST_LOG_LEVEL: $(yaml-quote ${KUBEPROXY_TEST_LOG_LEVEL})
EOF
    fi
  fi
  if [[ "${OS_DISTRIBUTION}" == "coreos" ]]; then
    # CoreOS-only env vars. TODO(yifan): Make them available on other distros.
    cat >>$file <<EOF
KUBERNETES_CONTAINER_RUNTIME: $(yaml-quote ${CONTAINER_RUNTIME:-docker})
RKT_VERSION: $(yaml-quote ${RKT_VERSION:-})
RKT_PATH: $(yaml-quote ${RKT_PATH:-})
KUBERNETES_CONFIGURE_CBR0: $(yaml-quote ${KUBERNETES_CONFIGURE_CBR0:-true})
EOF
  fi
}
