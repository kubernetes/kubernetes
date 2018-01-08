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

# This is an example script that starts vtctld.

set -e

script_root=`dirname "${BASH_SOURCE}"`
source $script_root/env.sh

service_type=${VTCTLD_SERVICE_TYPE:-'ClusterIP'}
cell=(`echo $CELLS | tr ',' ' '`) # ref to cell will get first element
TEST_MODE=${TEST_MODE:-'0'}

test_flags=`[[ $TEST_MODE -gt 0 ]] && echo '-enable_queries' || echo ''`

echo "Creating vtctld $service_type service..."
sed_script=""
for var in service_type; do
  sed_script+="s,{{$var}},${!var},g;"
done
cat vtctld-service-template.yaml | sed -e "$sed_script" | $KUBECTL $KUBECTL_OPTIONS create -f -

echo "Creating vtctld replicationcontroller..."
# Expand template variables
sed_script=""
for var in vitess_image backup_flags test_flags cell; do
  sed_script+="s,{{$var}},${!var},g;"
done

# Instantiate template and send to kubectl.
cat vtctld-controller-template.yaml | sed -e "$sed_script" | $KUBECTL $KUBECTL_OPTIONS create -f -

echo
echo "To access vtctld web UI, start kubectl proxy in another terminal:"
echo "  kubectl proxy --port=8001"
echo "Then visit http://localhost:8001/api/v1/proxy/namespaces/$VITESS_NAME/services/vtctld:web/"

