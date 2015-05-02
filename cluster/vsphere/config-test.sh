#!/bin/bash

# Copyright 2014 The Kubernetes Authors All rights reserved.
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

NUM_MINIONS=2
DISK=./kube/kube.vmdk
GUEST_ID=debian7_64Guest

INSTANCE_PREFIX="e2e-test-${USER}"
MASTER_TAG="${INSTANCE_PREFIX}-master"
MINION_TAG="${INSTANCE_PREFIX}-minion"

MASTER_NAME="${INSTANCE_PREFIX}-master"
MASTER_MEMORY_MB=1024
MASTER_CPU=1

MINION_NAMES=($(eval echo ${INSTANCE_PREFIX}-minion-{1..${NUM_MINIONS}}))
MINION_IP_RANGES=($(eval echo "10.244.{1..${NUM_MINIONS}}.0/24"))
MINION_MEMORY_MB=1024
MINION_CPU=1

PORTAL_NET="10.244.240.0/20"
