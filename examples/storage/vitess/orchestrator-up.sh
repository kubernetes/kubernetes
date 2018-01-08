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

# This is an example script that starts orchestrator.

set -e

script_root=`dirname "${BASH_SOURCE}"`
source $script_root/env.sh

echo "Creating orchestrator service, configmap, and replicationcontroller..."
sed_script=""
for var in service_type; do
  sed_script+="s,{{$var}},${!var},g;"
done

# Create configmap from orchestrator docker config file
$KUBECTL $KUBECTL_OPTIONS create configmap orchestrator-conf --from-file="${script_root}/../../docker/orchestrator/orchestrator.conf.json"

cat orchestrator-template.yaml | sed -e "$sed_script" | $KUBECTL $KUBECTL_OPTIONS create -f -

echo
echo "To access orchestrator web UI, start kubectl proxy in another terminal:"
echo "  kubectl proxy --port=8001"
echo "Then visit http://localhost:8001/api/v1/proxy/namespaces/$VITESS_NAME/services/orchestrator/"
