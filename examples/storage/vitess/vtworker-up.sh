#!/bin/bash

# Copyright 2017 Google Inc.
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

set -e

script_root=`dirname "${BASH_SOURCE}"`
source $script_root/env.sh

cell=(`echo $CELLS | tr ',' ' '`) # ref to cell will get first element
port=15032
grpc_port=15033

sed_script=""
for var in vitess_image cell port grpc_port; do
  sed_script+="s,{{$var}},${!var},g;"
done

echo "Creating vtworker pod in cell $cell..."
cat vtworker-controller-interactive-template.yaml | sed -e "$sed_script" | $KUBECTL $KUBECTL_OPTIONS create -f -

set +e

service_type='LoadBalancer'
echo "Creating vtworker $service_type service..."
sed_script=""
for var in service_type port grpc_port; do
  sed_script+="s,{{$var}},${!var},g;"
done
cat vtworker-service-template.yaml | sed -e "$sed_script" | $KUBECTL $KUBECTL_OPTIONS create -f -
