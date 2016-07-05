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


source "${KUBE_ROOT}/cluster/aws/common/common.sh"

SSH_USER=ubuntu

# Detects the AMI to use for ubuntu (considering the region)
#
# Vars set:
#   AWS_IMAGE
function detect-wily-image () {
  # This is the ubuntu 15.10 image for <region>, amd64, hvm:ebs-ssd
  # See here: http://cloud-images.ubuntu.com/locator/ec2/ for other images
  # This will need to be updated from time to time as amis are deprecated
  if [[ -z "${AWS_IMAGE-}" ]]; then
    case "${AWS_REGION}" in
      ap-northeast-1)
        AWS_IMAGE=ami-3355505d
        ;;

      ap-northeast-2)
        AWS_IMAGE=ami-e427e98a
        ;;

      ap-southeast-1)
        AWS_IMAGE=ami-60975903
        ;;

      eu-central-1)
        AWS_IMAGE=ami-6da2ba01
        ;;

      eu-west-1)
        AWS_IMAGE=ami-36a71645
        ;;

      sa-east-1)
        AWS_IMAGE=ami-fd36b691
        ;;

      us-east-1)
        AWS_IMAGE=ami-6610390c
        ;;

      us-west-1)
        AWS_IMAGE=ami-6e64120e
        ;;

      cn-north-1)
        AWS_IMAGE=ami-17a76f7a
        ;;

      us-gov-west-1)
        AWS_IMAGE=ami-b0bad893
        ;;

      ap-southeast-2)
        AWS_IMAGE=ami-3895b15b
        ;;

      us-west-2)
        AWS_IMAGE=ami-d95abcb9
        ;;

      *)
        echo "Please specify AWS_IMAGE directly (region ${AWS_REGION} not recognized)"
        exit 1
    esac
  fi
}

