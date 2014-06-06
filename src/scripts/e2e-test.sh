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

# Starts a Kubernetes cluster, verifies it can do basic things, and shuts it
# down.

# Exit on error
set -e

# Use testing config
export KUBE_CONFIG_FILE="config-test.sh"
source $(dirname $0)/util.sh

# Build a release
$(dirname $0)/../release/release.sh

# Now bring a test cluster up with that release.
$(dirname $0)/kube-up.sh

# Auto shutdown cluster when we exit
function shutdown-test-cluster () {
  echo "Shutting down test cluster in background."
  $(dirname $0)/kube-down.sh > /dev/null &
}
trap shutdown-test-cluster EXIT

# Launch a container
$(dirname $0)/cloudcfg.sh -p 8080:80 run dockerfile/nginx 2 myNginx

# Get minion IP addresses
detect-minions

# Verify that something is listening (nginx should give us a 404)
for (( i=0; i<${#KUBE_MINION_IP_ADDRESSES[@]}; i++)); do
  IP_ADDRESS=${KUBE_MINION_IP_ADDRESSES[$i]}
  echo "Trying to reach nginx instance that should be running at ${IP_ADDRESS}:8080..."
  curl "http://${IP_ADDRESS}:8080"
done

