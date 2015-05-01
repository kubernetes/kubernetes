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

# This script should be sourced as a part of config-test or config-default.
# Specifically, the following environment variables are assumed:
# - CLUSTER_NAME  (the name of the cluster)

MASTER_NAME="k8s-${CLUSTER_NAME}-master"
ZONE="${ZONE:-us-central1-f}"
NUM_MINIONS="${NUM_MINIONS:-2}"
MINION_NAMES=($(eval echo "k8s-${CLUSTER_NAME}-node-{1..${NUM_MINIONS}}"))
CLUSTER_API_VERSION="${CLUSTER_API_VERSION:-}"
# TODO(mbforbes): Actually plumb this through; this currently only works
#                 because we use the 'default' network by default.
NETWORK="${NETWORK:-default}"
NETWORK_RANGE="${NETWORK_RANGE:-10.240.0.0/16}"
FIREWALL_SSH="${FIREWALL_SSH:-${NETWORK}-allow-ssh}"
GCLOUD="${GCLOUD:-gcloud}"
GCLOUD_CONFIG_DIR="${GCLOUD_CONFIG_DIR:-${HOME}/.config/gcloud/kubernetes}"
ENABLE_CLUSTER_DNS=false

# This is a hack, but I keep setting this when I run commands manually, and
# then things grossly fail during normal runs because cluster/kubecfg.sh and
# cluster/kubectl.sh both use this if it's set.
unset KUBERNETES_MASTER
