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

# This script should be sourced as a part of config-test or config-default.
# Specifically, the following environment variables are assumed:
# - CLUSTER_NAME  (the name of the cluster)

ZONE="${ZONE:-us-central1-f}"
NUM_NODES="${NUM_NODES:-3}"
ADDITIONAL_ZONES="${ADDITIONAL_ZONES:-}"
CLUSTER_API_VERSION="${CLUSTER_API_VERSION:-}"
NETWORK="${NETWORK:-default}"
FIREWALL_SSH="${FIREWALL_SSH:-${NETWORK}-allow-ssh}"
GCLOUD="${GCLOUD:-gcloud}"
CMD_GROUP="${CMD_GROUP:-}"
GCLOUD_CONFIG_DIR="${GCLOUD_CONFIG_DIR:-${HOME}/.config/gcloud/kubernetes}"
MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-2}"
IMAGE_TYPE="${IMAGE_TYPE:-}"
if [[ "${FEDERATION:-}" == true ]]; then
    NODE_SCOPES="${NODE_SCOPES:-compute-rw,storage-ro,https://www.googleapis.com/auth/ndev.clouddns.readwrite}"
else
    NODE_SCOPES="${NODE_SCOPES:-compute-rw,storage-ro}"
fi

# WARNING: any new vars added here must correspond to options that can be
# passed to `gcloud {CMD_GROUP} container clusters create`, or they will
# have no effect. If you change/add a var used to toggle a value in
# cluster/gce/configure-vm.sh, please ping someone on GKE.

# This is a hack, but I keep setting this when I run commands manually, and
# then things grossly fail during normal runs because cluster/kubecfg.sh and
# cluster/kubectl.sh both use this if it's set.
unset KUBERNETES_MASTER
