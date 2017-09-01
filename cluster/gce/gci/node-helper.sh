#!/bin/bash

# Copyright 2016 The Kubernetes Authors.
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

# A library of helper functions and constant for GCI distro
source "${KUBE_ROOT}/cluster/gce/gci/helper.sh"

function get-node-instance-metadata {
  local metadata=""
  metadata+="kube-env=${KUBE_TEMP}/node-kube-env.yaml,"
  metadata+="user-data=${KUBE_ROOT}/cluster/gce/gci/node.yaml,"
  metadata+="configure-sh=${KUBE_ROOT}/cluster/gce/gci/configure.sh,"
  metadata+="cluster-name=${KUBE_TEMP}/cluster-name.txt,"
  metadata+="gci-update-strategy=${KUBE_TEMP}/gci-update.txt,"
  metadata+="gci-ensure-gke-docker=${KUBE_TEMP}/gci-ensure-gke-docker.txt,"
  metadata+="gci-docker-version=${KUBE_TEMP}/gci-docker-version.txt"
  echo "${metadata}"
}

# $1: template name (required).
function create-node-instance-template {
  local template_name="$1"
  ensure-gci-metadata-files
  create-node-template "$template_name" "${scope_flags[*]}" "$(get-node-instance-metadata)"
}
