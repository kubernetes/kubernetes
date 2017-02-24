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

# The following are default-specific settings.
CLUSTER_NAME="${CLUSTER_NAME:-${USER}-gke}"
NETWORK=${KUBE_GKE_NETWORK:-default}

# For ease of maintenance, extract any pieces that do not vary between default
# and test in a common config.
source $(dirname "${BASH_SOURCE}")/config-common.sh

# Optional: Install node logging
ENABLE_NODE_LOGGING=false
LOGGING_DESTINATION=gcp # options: elasticsearch, gcp

# Optional: When set to true, Elasticsearch and Kibana will be setup as part of the cluster bring up.
ENABLE_CLUSTER_LOGGING=false
ELASTICSEARCH_LOGGING_REPLICAS=1

# Optional: Deploy a L7 loadbalancer controller to fulfill Ingress requests:
#   glbc           - CE L7 Load Balancer Controller
ENABLE_L7_LOADBALANCING="${KUBE_ENABLE_L7_LOADBALANCING:-glbc}"

# Optional: Cluster monitoring to setup as part of the cluster bring up:
#   none     - No cluster monitoring setup
#   influxdb - Heapster, InfluxDB, and Grafana
#   google   - Heapster, Google Cloud Monitoring, and Google Cloud Logging
#   standalone     - Heapster only. Metrics available via Heapster REST API.
ENABLE_CLUSTER_MONITORING="${KUBE_ENABLE_CLUSTER_MONITORING:-standalone}"

KUBE_DELETE_NETWORK=${KUBE_DELETE_NETWORK:-false}
