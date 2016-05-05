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

# The configuration is based on upstart, which is in Ubuntu up to 14.04 LTS (Trusty).
# Ubuntu 15.04 and above replaced upstart with systemd as the init system.
# Consequently, the configuration cannot work on these images. In release-1.2 branch,
# GCI and Trusty share the configuration code. We have to keep the GCI specific code
# here as long as the release-1.2 branch has not been deprecated.

# Creates the GCI specific metadata files if they do not exit.
# Assumed var
#   KUBE_TEMP
function ensure-gci-metadata-files {
  if [[ ! -f "${KUBE_TEMP}/gci-update.txt" ]]; then
    cat >"${KUBE_TEMP}/gci-update.txt" << EOF
update_disabled
EOF
  fi
  if [[ ! -f "${KUBE_TEMP}/gci-docker.txt" ]]; then
    cat >"${KUBE_TEMP}/gci-docker.txt" << EOF
true
EOF
  fi
}

# $1: template name (required)
function create-node-instance-template {
  local template_name="$1"
  if [[ "${OS_DISTRIBUTION}" == "gci" && "${NODE_IMAGE}" == gci* ]]; then
    ensure-gci-metadata-files
    create-node-template "$template_name" "${scope_flags[*]}" \
      "kube-env=${KUBE_TEMP}/node-kube-env.yaml" \
      "user-data=${KUBE_ROOT}/cluster/gce/trusty/node.yaml" \
      "configure-sh=${KUBE_ROOT}/cluster/gce/trusty/configure.sh" \
      "cluster-name=${KUBE_TEMP}/cluster-name.txt" \
      "gci-update-strategy=${KUBE_TEMP}/gci-update.txt" \
      "gci-ensure-gke-docker=${KUBE_TEMP}/gci-docker.txt"
  else
    create-node-template "$template_name" "${scope_flags[*]}" \
      "kube-env=${KUBE_TEMP}/node-kube-env.yaml" \
      "user-data=${KUBE_ROOT}/cluster/gce/trusty/node.yaml" \
      "configure-sh=${KUBE_ROOT}/cluster/gce/trusty/configure.sh" \
      "cluster-name=${KUBE_TEMP}/cluster-name.txt"
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
  local image_metadata=""
  if [[ "${OS_DISTRIBUTION}" == "gci" && "${MASTER_IMAGE}" == gci* ]]; then
    ensure-gci-metadata-files
    image_metadata=",gci-update-strategy=${KUBE_TEMP}/gci-update.txt,gci-ensure-gke-docker=${KUBE_TEMP}/gci-docker.txt"
  fi

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
      "kube-env=${KUBE_TEMP}/master-kube-env.yaml,user-data=${KUBE_ROOT}/cluster/gce/trusty/master.yaml,configure-sh=${KUBE_ROOT}/cluster/gce/trusty/configure.sh,cluster-name=${KUBE_TEMP}/cluster-name.txt${image_metadata}" \
    --disk "name=${MASTER_NAME}-pd,device-name=master-pd,mode=rw,boot=no,auto-delete=no"
}
