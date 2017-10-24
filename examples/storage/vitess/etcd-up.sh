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

# This is an example script that creates etcd clusters.
# Vitess requires a global cluster, as well as one for each cell.
#
# For automatic discovery, an etcd cluster can be bootstrapped from an
# existing cluster. In this example, we use an externally-run discovery
# service, but you can use your own. See the etcd docs for more:
# https://github.com/coreos/etcd/blob/v2.0.13/Documentation/clustering.md

set -e

script_root=`dirname "${BASH_SOURCE}"`
source $script_root/env.sh

replicas=${ETCD_REPLICAS:-3}
cells=`echo $CELLS | tr ',' ' '`

for cell in 'global' $cells; do
  # Create the client service, which will load-balance across all replicas.
  echo "Creating etcd service for $cell cell..."
  cat etcd-service-template.yaml | \
    sed -e "s/{{cell}}/$cell/g" | \
    $KUBECTL $KUBECTL_OPTIONS create -f -

  # Expand template variables
  sed_script=""
  for var in cell replicas; do
    sed_script+="s,{{$var}},${!var},g;"
  done

  # Create the replication controller.
  echo "Creating etcd replicationcontroller for $cell cell..."
  cat etcd-controller-template.yaml | sed -e "$sed_script" | $KUBECTL $KUBECTL_OPTIONS create -f -
done

