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
        AWS_IMAGE=ami-907fa690
        ;;

      ap-southeast-1)
        AWS_IMAGE=ami-b4a79de6
        ;;

      eu-central-1)
        AWS_IMAGE=ami-e8635bf5
        ;;

      eu-west-1)
        AWS_IMAGE=ami-0fd0ae78
        ;;

      sa-east-1)
        AWS_IMAGE=ami-f9f675e4
        ;;

      us-east-1)
        AWS_IMAGE=ami-f57b8f9e
        ;;

      us-west-1)
        AWS_IMAGE=ami-87b643c3
        ;;

      cn-north-1)
        AWS_IMAGE=ami-3abf2203
        ;;

      #us-gov-west-1)
      #  AWS_IMAGE=?Not available?
      #  ;;

      ap-southeast-2)
        AWS_IMAGE=ami-1bb9c221
        ;;

      us-west-2)
        AWS_IMAGE=ami-33566d03
        ;;

      *)
        echo "Please specify AWS_IMAGE directly (region not recognized)"
        exit 1
    esac
  fi
}

