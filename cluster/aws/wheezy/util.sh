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


# A library of helper functions for Wheezy.

source "${KUBE_ROOT}/cluster/aws/trusty/common.sh"

SSH_USER=admin

# Detects the AMI to use for wheezy (considering the region)
# Source: https://wiki.debian.org/Cloud/AmazonEC2Image/Wheezy
#
# Vars set:
#   AWS_IMAGE
function detect-wheezy-image () {
  if [[ -z "${AWS_IMAGE-}" ]]; then
    case "${AWS_REGION}" in
      ap-northeast-1)
        AWS_IMAGE=ami-b25d44b3
        ;;

      ap-southeast-1)
        AWS_IMAGE=ami-aeb49ffc
        ;;

      ap-southeast-2)
        AWS_IMAGE=ami-6b770351
        ;;

      eu-central-1)
        AWS_IMAGE=ami-98043785
        ;;

      eu-west-1)
        AWS_IMAGE=ami-61e56916
        ;;

      sa-east-1)
        AWS_IMAGE=ami-3d8b3720
        ;;

      us-east-1)
        AWS_IMAGE=ami-e0efab88
        ;;

      us-west-1)
        AWS_IMAGE=ami-b4869ff1
        ;;

      us-west-2)
        AWS_IMAGE=ami-431a4273
        ;;

      us-gov-west-1)
        AWS_IMAGE=ami-d13455f2
        ;;

      cn-north-1)
        AWS_IMAGE=ami-48029071
        ;;

      *)
        echo "Please specify AWS_IMAGE directly (region ${AWS_REGION} not recognized)"
        exit 1
    esac
  fi
}
