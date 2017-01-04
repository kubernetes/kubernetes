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

# A library of helper functions and constant for the Container Linux distro.
source "${KUBE_ROOT}/cluster/gce/container-linux/helper.sh"

# $1: template name (required).
function create-node-instance-template {
  local template_name="$1"

  create-node-template "$template_name" "${scope_flags[*]}" \
    "kube-env=${KUBE_TEMP}/node-kube-env.yaml" \
    "user-data=${KUBE_ROOT}/cluster/gce/container-linux/node.yaml" \
    "configure-sh=${KUBE_ROOT}/cluster/gce/container-linux/configure.sh" \
    "cluster-name=${KUBE_TEMP}/cluster-name.txt"
  # TODO(euank): We should include update-strategy here. We should also switch to ignition
}
