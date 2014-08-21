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

# Launches an nginx container and verifies it can be reached. Assumes that
# we're being called by hack/e2e-test.sh (we use some env vars it sets up).

# Exit on error
set -e

source "${KUBE_REPO_ROOT}/cluster/kube-env.sh"
source "${KUBE_REPO_ROOT}/cluster/$KUBERNETES_PROVIDER/util.sh"

function validate() {
    POD_ID_LIST=$($CLOUDCFG '-template={{range.Items}}{{.ID}} {{end}}' -l name=$controller list pods)
    # Container turn up on a clean cluster can take a while for the docker image pull.
    ALL_RUNNING=0
    while [ $ALL_RUNNING -ne 1 ]; do
	echo "Waiting for all containers in pod to come up."
	sleep 5
	ALL_RUNNING=1
	for id in $POD_ID_LIST; do
	    CURRENT_STATUS=$($CLOUDCFG -template '{{and .CurrentState.Info.datacontroller.State.Running .CurrentState.Info.net.State.Running}}' get pods/$id)
	    if [ "$CURRENT_STATUS" != "true" ]; then
		ALL_RUNNING=0
	    fi
	done
    done

    ids=($POD_ID_LIST)
    if [ ${#ids[@]} -ne $1 ]; then
	echo "Unexpected number of pods: ${#ids[@]}.  Expected $1"
	exit 1
    fi
}

controller=dataController

# Launch a container
$CLOUDCFG -p 8080:80 run brendanburns/data 2 $controller

function teardown() {
  echo "Cleaning up test artifacts"
  $CLOUDCFG stop $controller
  $CLOUDCFG rm $controller
}

trap "teardown" EXIT

validate 2

$CLOUDCFG resize $controller 1
sleep 2
validate 1

$CLOUDCFG resize $controller 2
sleep 2
validate 2

# TODO: test rolling update here, but to do so, we need to make the update blocking
# $CLOUDCFG -u=20s rollingupdate $controller
#
# Wait for the replica controller to recreate
# sleep 10
#
# validate 2

exit 0
