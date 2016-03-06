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
#
# Vars set:
#   AWS_IMAGE
function detect-jessie-image () {
  if [[ -z "${AWS_IMAGE-}" ]]; then
    # TODO: publish on a k8s AWS account
    aws_account="721322707521"
    # TODO: we could use tags for the image
    if [[ -z "${AWS_IMAGE_NAME:-}" ]]; then
      AWS_IMAGE_NAME="k8s-1.2-debian-jessie-amd64-hvm-2016-03-05-ebs"
    fi
    AWS_IMAGE=`aws ec2 describe-images --owner ${aws_account} --filters Name=name,Values=${AWS_IMAGE_NAME} --query Images[].ImageId --output text`
    if [[ -z "${AWS_IMAGE-}" ]]; then
      echo "Please specify AWS_IMAGE directly (image ${AWS_IMAGE_NAME} not found in region ${AWS_REGION})"
      exit 1
    fi
  fi
}
