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

# A library of helper functions and constant for debian os distro

# $1: template name (required)
function create-node-instance-template {
  local template_name="$1"
  prepare-startup-script
  create-node-template "$template_name" "${scope_flags}" \
    "startup-script=${KUBE_TEMP}/configure-vm.sh" \
    "kube-env=${KUBE_TEMP}/node-kube-env.yaml" \
    "cluster-name=${KUBE_TEMP}/cluster-name.txt"
}
