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

source "$(dirname "${BASH_SOURCE}")/../config-default.sh"
KUBE_APISERVER_SECURE_PORT=6443

# Optional: Install node monitoring.
# TODO Enable node monitoring for CoreOS-based clusters.
ENABLE_NODE_MONITORING="false"

# Optional: When set to true, heapster will be setup as part of the cluster bring up.
# TODO Enable cluster monitoring for CoreOS-based clusters.
ENABLE_CLUSTER_MONITORING="false"

# Optional: Enable node logging.
# TODO: Enable logging for CoreOS-based clusters.
ENABLE_NODE_LOGGING="false"

# Optional: When set to true, Elasticsearch and Kibana will be setup as part of the cluster bring up.
ENABLE_CLUSTER_LOGGING="false"

# Don't require https for registries in our local RFC1918 network
EXTRA_DOCKER_OPTS="--insecure-registry 10.0.0.0/8"

# Optional: Install cluster DNS.
# TODO: Enable cluster DNS for CoreOS-based clusters.
ENABLE_CLUSTER_DNS=false

