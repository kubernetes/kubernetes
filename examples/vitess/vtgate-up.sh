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

# This is an example script that starts a vtgate replicationcontroller.

set -e

script_root=`dirname "${BASH_SOURCE}"`
source $script_root/env.sh

VTGATE_REPLICAS=${VTGATE_REPLICAS:-3}
VTGATE_TEMPLATE=${VTGATE_TEMPLATE:-'vtgate-controller-template.yaml'}

replicas=$VTGATE_REPLICAS

echo "Creating vtgate service..."
$KUBECTL create -f vtgate-service.yaml

sed_script=""
for var in replicas; do
  sed_script+="s,{{$var}},${!var},g;"
done

echo "Creating vtgate replicationcontroller..."
cat $VTGATE_TEMPLATE | sed -e "$sed_script" | $KUBECTL create -f -
