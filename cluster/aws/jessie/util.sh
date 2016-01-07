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


# A library of helper functions for Jessie.

source "${KUBE_ROOT}/cluster/aws/trusty/common.sh"

SSH_USER=admin

# Detects the AMI to use for jessie (considering the region)
# Source: https://wiki.debian.org/Cloud/AmazonEC2Image/Jessie
#
# Vars set:
#   AWS_IMAGE
function detect-jessie-image () {
  if [[ -z "${AWS_IMAGE-}" ]]; then
    case "${AWS_REGION}" in
      ap-northeast-1)
        AWS_IMAGE=ami-e624fbe6
        ;;

      ap-southeast-1)
        AWS_IMAGE=ami-ac360cfe
        ;;

      ap-southeast-2)
        AWS_IMAGE=ami-bbc5bd81
        ;;

      eu-central-1)
        AWS_IMAGE=ami-02b78e1f
        ;;

      eu-west-1)
        AWS_IMAGE=ami-e31a6594
        ;;

      sa-east-1)
        AWS_IMAGE=ami-0972f214
        ;;

      us-east-1)
        AWS_IMAGE=ami-116d857a
        ;;

      us-west-1)
        AWS_IMAGE=ami-05cf2541
        ;;

      us-west-2)
        AWS_IMAGE=ami-818eb7b1
        ;;

      cn-north-1)
        AWS_IMAGE=ami-888815b1
        ;;

      us-gov-west-1)
        AWS_IMAGE=ami-35b5d516
        ;;

      *)
        echo "Please specify AWS_IMAGE directly (region ${AWS_REGION} not recognized)"
        exit 1
    esac
  fi
}
