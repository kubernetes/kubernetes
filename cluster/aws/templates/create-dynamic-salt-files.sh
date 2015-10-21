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
node_instance_prefix: '$(echo "$NODE_INSTANCE_PREFIX" | sed -e "s/'/''/g")'
cluster_cidr: '$(echo "$CLUSTER_IP_RANGE" | sed -e "s/'/''/g")'
allocate_node_cidrs: '$(echo "$ALLOCATE_NODE_CIDRS" | sed -e "s/'/''/g")'
service_cluster_ip_range: '$(echo "$SERVICE_CLUSTER_IP_RANGE" | sed -e "s/'/''/g")'
enable_cluster_monitoring: '$(echo "$ENABLE_CLUSTER_MONITORING" | sed -e "s/'/''/g")'
enable_cluster_logging: '$(echo "$ENABLE_CLUSTER_LOGGING" | sed -e "s/'/''/g")'
enable_cluster_ui: '$(echo "$ENABLE_CLUSTER_UI" | sed -e "s/'/''/g")'
enable_node_logging: '$(echo "$ENABLE_NODE_LOGGING" | sed -e "s/'/''/g")'
logging_destination: '$(echo "$LOGGING_DESTINATION" | sed -e "s/'/''/g")'
elasticsearch_replicas: '$(echo "$ELASTICSEARCH_LOGGING_REPLICAS" | sed -e "s/'/''/g")'
enable_cluster_dns: '$(echo "$ENABLE_CLUSTER_DNS" | sed -e "s/'/''/g")'
dns_replicas: '$(echo "$DNS_REPLICAS" | sed -e "s/'/''/g")'
dns_server: '$(echo "$DNS_SERVER_IP" | sed -e "s/'/''/g")'
dns_domain: '$(echo "$DNS_DOMAIN" | sed -e "s/'/''/g")'
admission_control: '$(echo "$ADMISSION_CONTROL" | sed -e "s/'/''/g")'
num_nodes: $(echo "${NUM_MINIONS}")
EOF

readonly BASIC_AUTH_FILE="/srv/salt-overlay/salt/kube-apiserver/basic_auth.csv"
if [ ! -e "${BASIC_AUTH_FILE}" ]; then
  mkdir -p /srv/salt-overlay/salt/kube-apiserver
  (umask 077;
    echo "${KUBE_PASSWORD},${KUBE_USER},admin" > "${BASIC_AUTH_FILE}")
fi

# Generate and distribute a shared secret (bearer token) to
# apiserver and the nodes so that kubelet and kube-proxy can
# authenticate to apiserver.
kubelet_token=$KUBELET_TOKEN
kube_proxy_token=$KUBE_PROXY_TOKEN

# Make a list of tokens and usernames to be pushed to the apiserver
mkdir -p /srv/salt-overlay/salt/kube-apiserver
readonly KNOWN_TOKENS_FILE="/srv/salt-overlay/salt/kube-apiserver/known_tokens.csv"
(umask u=rw,go= ; echo "$kubelet_token,kubelet,kubelet" > $KNOWN_TOKENS_FILE ;
echo "$kube_proxy_token,kube_proxy,kube_proxy" >> $KNOWN_TOKENS_FILE)

mkdir -p /srv/salt-overlay/salt/kubelet
kubelet_auth_file="/srv/salt-overlay/salt/kubelet/kubernetes_auth"
(umask u=rw,go= ; echo "{\"BearerToken\": \"$kubelet_token\", \"Insecure\": true }" > $kubelet_auth_file)

mkdir -p /srv/salt-overlay/salt/kube-proxy
kube_proxy_kubeconfig_file="/srv/salt-overlay/salt/kube-proxy/kubeconfig"
cat > "${kube_proxy_kubeconfig_file}" <<EOF
apiVersion: v1
kind: Config
users:
- name: kube-proxy
  user:
    token: ${kube_proxy_token}
clusters:
- name: local
  cluster:
     insecure-skip-tls-verify: true
contexts:
- context:
    cluster: local
    user: kube-proxy
  name: service-account-context
current-context: service-account-context
EOF

mkdir -p /srv/salt-overlay/salt/kubelet
kubelet_kubeconfig_file="/srv/salt-overlay/salt/kubelet/kubeconfig"
cat > "${kubelet_kubeconfig_file}" <<EOF
apiVersion: v1
kind: Config
users:
- name: kubelet
  user:
    token: ${kubelet_token}
clusters:
- name: local
  cluster:
     insecure-skip-tls-verify: true
contexts:
- context:
    cluster: local
    user: kubelet
  name: service-account-context
current-context: service-account-context
EOF

# Generate tokens for other "service accounts".  Append to known_tokens.
#
# NB: If this list ever changes, this script actually has to
# change to detect the existence of this file, kill any deleted
# old tokens and add any new tokens (to handle the upgrade case).
service_accounts=("system:scheduler" "system:controller_manager" "system:logging" "system:monitoring" "system:dns")
for account in "${service_accounts[@]}"; do
  token=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
  echo "${token},${account},${account}" >> "${KNOWN_TOKENS_FILE}"
done
