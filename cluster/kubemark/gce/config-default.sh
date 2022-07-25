#!/usr/bin/env bash

# Copyright 2017 The Kubernetes Authors.
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

# A configuration for Kubemark cluster. It doesn't need to be kept in
# sync with gce/config-default.sh (except the filename, because I'm reusing
# gce/util.sh script which assumes config filename), but if some things that
# are enabled by default should not run in hollow clusters, they should be disabled here.

# shellcheck disable=SC2034 # Variables sourced in other scripts.

source "${KUBE_ROOT}/cluster/gce/config-common.sh"

CLEANUP_KUBEMARK_IMAGE=${CLEANUP_KUBEMARK_IMAGE:-true}
TEST_CLUSTER_LOG_LEVEL="${TEST_CLUSTER_LOG_LEVEL:-"--v=4"}"

# If you want to set up multiple kubemark clusters with different "names",
# you should change this env per each start-kubemark.sh invocation.
KUBE_GCE_INSTANCE_PREFIX="${KUBE_GCE_INSTANCE_PREFIX:-"default"}"

# NUM_NODES is used by start-kubemark.sh to determine a correct number of replicas.
NUM_NODES=${KUBEMARK_NUM_NODES:-10}
NUM_WINDOWS_NODES=${KUBEMARK_NUM_WINDOWS_NODES:-0}

HOLLOW_KUBELET_TEST_LOG_LEVEL="${HOLLOW_KUBELET_TEST_LOG_LEVEL:-$TEST_CLUSTER_LOG_LEVEL}"
HOLLOW_PROXY_TEST_LOG_LEVEL="${HOLLOW_PROXY_TEST_LOG_LEVEL:-$TEST_CLUSTER_LOG_LEVEL}"

# Hollow-node components' test arguments.
HOLLOW_KUBELET_TEST_ARGS="${HOLLOW_KUBELET_TEST_ARGS:-} ${HOLLOW_KUBELET_TEST_LOG_LEVEL}"
HOLLOW_PROXY_TEST_ARGS="${HOLLOW_PROXY_TEST_ARGS:-} ${HOLLOW_PROXY_TEST_LOG_LEVEL}"

# Optional: Enable cluster autoscaler.
ENABLE_KUBEMARK_CLUSTER_AUTOSCALER="${ENABLE_KUBEMARK_CLUSTER_AUTOSCALER:-false}"
# When using Cluster Autoscaler, always start with one hollow-node replica.
# NUM_NODES should not be specified by the user. Instead we use
# NUM_NODES=KUBEMARK_AUTOSCALER_MAX_NODES. This gives other cluster components
# (e.g. kubemark master, Heapster) enough resources to handle maximum cluster size.
if [[ "${ENABLE_KUBEMARK_CLUSTER_AUTOSCALER}" == "true" ]]; then
  NUM_REPLICAS=1
  if [[ -n "$NUM_NODES" ]]; then
    echo "WARNING: Using Cluster Autoscaler, ignoring NUM_NODES parameter. Set KUBEMARK_AUTOSCALER_MAX_NODES to specify maximum size of the cluster."
  fi
fi

#Optional: Enable kube dns.
ENABLE_KUBEMARK_KUBE_DNS="${ENABLE_KUBEMARK_KUBE_DNS:-true}"
KUBE_DNS_DOMAIN="${KUBE_DNS_DOMAIN:-cluster.local}"

CLEANUP_KUBEMARK_IMAGE=false
