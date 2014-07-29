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

# Push a new release to the cluster.
#
# This will find the release tar, cause it to be downloaded, unpacked, installed
# and enacted.

# exit on any error
set -e

source $(dirname $0)/kube-env.sh
source $(dirname $0)/$KUBERNETES_PROVIDER/util.sh

echo "Updating cluster using provider: $KUBERNETES_PROVIDER"

verify-prereqs
kube-push

source $(dirname $0)/validate-cluster.sh

echo "Done"
