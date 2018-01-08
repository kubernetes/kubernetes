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

SHARDS=${SHARDS:-'-80,80-'}
TABLETS_PER_SHARD=${TABLETS_PER_SHARD:-3}
CELLS=${CELLS:-'test'}
TEST_MODE=${TEST_MODE:-'0'}
VITESS_NAME=${VITESS_NAME:-'vitess'}

export VITESS_NAME=$VITESS_NAME

./vtgate-down.sh
SHARDS=$SHARDS CELLS=$CELLS TABLETS_PER_SHARD=$TABLETS_PER_SHARD ./vttablet-down.sh
./vtctld-down.sh
./etcd-down.sh

if [ $TEST_MODE -gt 0 ]; then
  gcloud compute firewall-rules delete ${VITESS_NAME}-vtctld -q
fi

for cell in `echo $CELLS | tr ',' ' '`; do
  gcloud compute firewall-rules delete ${VITESS_NAME}-vtgate-$cell -q
done

./namespace-down.sh
