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

## Contains configuration values for interacting with the Vagrant cluster

# NUMBER OF MINIONS IN THE CLUSTER
NUM_MINIONS=1

# IP LOCATIONS FOR INTERACTING WITH THE MASTER
export KUBE_MASTER_IP="127.0.0.1"
export KUBERNETES_MASTER="http://127.0.0.1:8080"

# IP LOCATIONS FOR INTERACTING WITH THE MINIONS
for (( i=0; i <${NUM_MINIONS}; i++)) do
	KUBE_MINION_IP_ADDRESSES[$i]="127.0.0.1"
done
