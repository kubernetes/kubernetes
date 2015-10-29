#!/bin/bash

# Copyright 2014 The Kubernetes Authors All rights reserved.
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

# Create the overlay files for the salt tree.  We create these in a separate
# place so that we can blow away the rest of the salt configs on a kube-push and
# re-apply these.

mkdir -p /srv/salt-overlay/pillar
cat <<EOF >/srv/salt-overlay/pillar/cluster-params.sls
instance_prefix: '$(echo "$INSTANCE_PREFIX" | sed -e "s/'/''/g")'
node_instance_prefix: $NODE_INSTANCE_PREFIX
service_cluster_ip_range: $SERVICE_CLUSTER_IP_RANGE
enable_cluster_monitoring: $ENABLE_CLUSTER_MONITORING
enable_cluster_logging: $ENABLE_CLUSTER_LOGGING
enable_cluster_ui: $ENABLE_CLUSTER_UI
enable_node_logging: $ENABLE_NODE_LOGGING
logging_destination: $LOGGING_DESTINATION
elasticsearch_replicas: $ELASTICSEARCH_LOGGING_REPLICAS
enable_cluster_dns: $ENABLE_CLUSTER_DNS
dns_replicas: $DNS_REPLICAS
dns_server: $DNS_SERVER_IP
dns_domain: $DNS_DOMAIN
e2e_storage_test_environment: $E2E_STORAGE_TEST_ENVIRONMENT

EOF

mkdir -p /srv/salt-overlay/salt/nginx
echo $MASTER_HTPASSWD > /srv/salt-overlay/salt/nginx/htpasswd
