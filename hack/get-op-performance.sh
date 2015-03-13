#!/bin/bash

# Copyright 2015 Google Inc. All rights reserved.
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


set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
KUBECTL=${KUBE_ROOT}/cluster/kubectl.sh

OPERATION=get
RESOURCE=pod
NO_RC_PER_NODE=5
NO_PODS_PER_RC=1
POD_PREFIX=test
NO_REPS=10

NO_NODES=$((`$KUBECTL get node | wc -l` - 1))
NO_RC=$(($NO_NODES * $NO_RC_PER_NODE)) 
NO_PODS=$(($NO_RC * $NO_PODS_PER_RC))

create_real=()
create_user=()
remove_real=()
remove_user=()

function create_load {
	echo "Creating..."
	for i in `seq $NO_RC`; do $KUBECTL run-container ${POD_PREFIX}$i --image=kubernetes/pause --replicas=${NO_PODS_PER_RC} 2> /dev/null > /dev/null & done; time wait
}

function run_experiment {
	echo "Testing..."
	for i in `seq $NO_REPS`; do $KUBECTL $OPERATION $RESOURCE > /dev/null 2> /dev/null & time wait; done
}

function remove_load {
	echo "Removing..."
	for i in `seq $NO_RC`; do $KUBECTL stop rc ${POD_PREFIX}$i > /dev/null 2> /dev/null & done; time wait
}

create_load
run_experiment
remove_load