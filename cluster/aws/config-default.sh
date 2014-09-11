#!/bin/bash

# Copyright 2014 Google Inc. All rights reserved.
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

ZONE=eu-west-1
MASTER_SIZE=t2.small
MINION_SIZE=t2.small
NUM_MINIONS=4

IMAGE=ami-0307d674
INSTANCE_PREFIX=kubernetes
AWS_SSH_KEY=$HOME/.ssh/kube_aws_rsa

MASTER_NAME="ip-172-20-0-9.$ZONE.compute.internal"
MASTER_TAG="${INSTANCE_PREFIX}-master"
MINION_TAG="${INSTANCE_PREFIX}-minion"
MINION_NAMES=($(eval echo ip-172-20-0-1{0..$(($NUM_MINIONS-1))}.$ZONE.compute.internal))
MINION_IP_RANGES=($(eval echo "10.244.{1..${NUM_MINIONS}}.0/24"))
MINION_SCOPES=""
