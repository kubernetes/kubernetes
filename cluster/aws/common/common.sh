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


# A library of common helper functions for Ubuntus & Debians.

function detect-minion-image() {
  if [[ -z "${KUBE_NODE_IMAGE=-}" ]]; then
    detect-image
    KUBE_NODE_IMAGE=$AWS_IMAGE
  fi
}

function generate-minion-user-data {
  # We pipe this to the ami as a startup script in the user-data field.  Requires a compatible ami
  echo "#! /bin/bash"
  echo "SALT_MASTER='${MASTER_INTERNAL_IP}'"
  echo "DOCKER_OPTS='${EXTRA_DOCKER_OPTS:-}'"
  echo "readonly NON_MASQUERADE_CIDR='${NON_MASQUERADE_CIDR:-}'"
  echo "readonly DOCKER_STORAGE='${DOCKER_STORAGE:-}'"
  grep -v "^#" "${KUBE_ROOT}/cluster/aws/templates/common.sh"
  grep -v "^#" "${KUBE_ROOT}/cluster/aws/templates/format-disks.sh"
  grep -v "^#" "${KUBE_ROOT}/cluster/aws/templates/salt-minion.sh"
}

function check-minion() {
  local minion_ip=$1

  local output=$(ssh -oStrictHostKeyChecking=no -i "${AWS_SSH_KEY}" ${SSH_USER}@$minion_ip sudo docker ps -a 2>/dev/null)
  if [[ -z "${output}" ]]; then
    ssh -oStrictHostKeyChecking=no -i "${AWS_SSH_KEY}" ${SSH_USER}@$minion_ip sudo service docker start > $LOG 2>&1
    echo "not working yet"
  else
    echo "working"
  fi
}
