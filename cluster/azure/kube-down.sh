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

# Tear down a Kubernetes cluster.
SCRIPT_DIR=$(CDPATH="" cd $(dirname $0); pwd)
source $SCRIPT_DIR/../../release/azure/config.sh
source $SCRIPT_DIR/../util.sh

echo "Bringing down cluster"
azure vm delete $MASTER_NAME -b -q
for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
    azure vm delete ${MINION_NAMES[$i]} -b -q
done
