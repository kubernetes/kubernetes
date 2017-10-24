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

# This is an example script that runs vtworker.

set -e

script_root=`dirname "${BASH_SOURCE}"`
source $script_root/env.sh

cell=test
port=15003
vtworker_command="$*"

# Expand template variables
sed_script=""
for var in vitess_image cell port vtworker_command; do
  sed_script+="s,{{$var}},${!var},g;"
done

# Instantiate template and send to kubectl.
echo "Creating vtworker pod in cell $cell..."
cat vtworker-pod-template.yaml | sed -e "$sed_script" | $KUBECTL $KUBECTL_OPTIONS create -f -

set +e

# Wait for vtworker pod to show up.
until [ $($KUBECTL $KUBECTL_OPTIONS get pod -o template --template '{{.status.phase}}' vtworker 2> /dev/null) = "Running" ]; do
  echo "Waiting for vtworker pod to be created..."
	sleep 1
done

echo "Following vtworker logs until termination..."
$KUBECTL $KUBECTL_OPTIONS logs -f vtworker

# Get vtworker exit code. Wait for complete shutdown.
# (Although logs -f exited, the pod isn't fully shutdown yet and the exit code is not available yet.)
until [ $($KUBECTL $KUBECTL_OPTIONS get pod -o template --template '{{.status.phase}}' vtworker 2> /dev/null) != "Running" ]; do
  echo "Waiting for vtworker pod to shutdown completely..."
  sleep 1
done
exit_code=$($KUBECTL $KUBECTL_OPTIONS get -o template --template '{{(index .status.containerStatuses 0).state.terminated.exitCode}}' pods vtworker)

echo "Deleting vtworker pod..."
$KUBECTL $KUBECTL_OPTIONS delete pod vtworker

if [ "$exit_code" != "0" ]; then
  echo
  echo "ERROR: vtworker did not run successfully (return value: $exit_code). Please check the error log above and try to run it again."
  exit 1
fi
