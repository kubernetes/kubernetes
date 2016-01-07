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


source "${KUBE_ROOT}/cluster/aws/trusty/common.sh"

SSH_USER=ubuntu


# Detects the AMI to use for ubuntu (considering the region)
#
# Vars set:
#   AWS_IMAGE
function detect-vivid-image () {
  # This is the ubuntu 15.04 image for <region>, amd64, hvm:ebs-ssd
  # See here: http://cloud-images.ubuntu.com/locator/ec2/ for other images
  # This will need to be updated from time to time as amis are deprecated
  if [[ -z "${AWS_IMAGE-}" ]]; then
    case "${AWS_REGION}" in
      ap-northeast-1)
        AWS_IMAGE=ami-56e28056
        ;;

      ap-southeast-1)
        AWS_IMAGE=ami-7e2a392c
        ;;

      eu-central-1)
        AWS_IMAGE=ami-58c1cd45
        ;;

      eu-west-1)
        AWS_IMAGE=ami-ef8ab698
        ;;

      sa-east-1)
        AWS_IMAGE=ami-c16afbdc
        ;;

      us-east-1)
        AWS_IMAGE=ami-c3772aa6
        ;;

      us-west-1)
        AWS_IMAGE=ami-6fbe7d2b
        ;;

      cn-north-1)
        AWS_IMAGE=ami-7a1b8743
        ;;

      #us-gov-west-1)
      #  AWS_IMAGE=?Not available?
      #  ;;

      ap-southeast-2)
        AWS_IMAGE=ami-0b93d931
        ;;

      us-west-2)
        AWS_IMAGE=ami-6e67845d
        ;;

      *)
        echo "Please specify AWS_IMAGE directly (region ${AWS_REGION} not recognized)"
        exit 1
    esac
  fi
}

