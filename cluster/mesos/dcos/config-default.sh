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

## Contains configuration values for interacting with the mesos/dcos cluster

NUM_NODES=${NUM_NODES:-2}
INSTANCE_PREFIX="${INSTANCE_PREFIX:-kubernetes}"
MASTER_NAME="${INSTANCE_PREFIX}-master"
NODE_NAMES=($(eval echo ${INSTANCE_PREFIX}-minion-{1..${NUM_NODES}}))

# Timeout (in seconds) to wait for each addon to come up
MESOS_DCOS_ADDON_TIMEOUT="${MESOS_DCOS_ADDON_TIMEOUT:-180}"
