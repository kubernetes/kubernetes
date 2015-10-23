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

# A library of helper functions and constant for ubuntu os distro

# The code and configuration is for running node instances on Ubuntu images.
# The master is still on Debian. In addition, the configuration is based on
# upstart, which is in Ubuntu upto 14.04 LTS (Trusty). Ubuntu 15.04 and above
# replaced upstart with systemd as the init system. Consequently, the
# configuration cannot work on these images.

# By sourcing debian's helper.sh, we use the same build-kube-env and
# create-master-instance functions as debian. But we overwrite the
# create-node-instance-template function to use Ubuntu.
source "${KUBE_ROOT}/cluster/gce/debian/helper.sh"

# $1: template name (required)
function create-node-instance-template {
  local template_name="$1"
  create-node-template "$template_name" "${scope_flags[*]}" \
		"kube-env=${KUBE_TEMP}/node-kube-env.yaml" \
    "user-data=${KUBE_ROOT}/cluster/gce/trusty/node.yaml"
}
