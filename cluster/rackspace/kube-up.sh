#!/bin/bash

# Copyright 2014 The Kubernetes Authors.
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

# Bring up a Kubernetes cluster.
#
# If the full release name (gs://<bucket>/<release>) is passed in then we take
# that directly.  If not then we assume we are doing development stuff and take
# the defaults in the release config.

# exit on any error
set -e

source $(dirname $0)/../kube-util.sh

echo "Starting cluster using provider: $KUBERNETES_PROVIDER"

verify-prereqs
kube-up

# skipping validation for now until since machines show up as private IPs
# source $(dirname $0)/validate-cluster.sh

echo "Done"
