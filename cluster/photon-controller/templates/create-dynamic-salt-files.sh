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

#generate token files

KUBELET_TOKEN=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
KUBE_PROXY_TOKEN=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
known_tokens_file="/srv/salt-overlay/salt/kube-apiserver/known_tokens.csv"
if [[ ! -f "${known_tokens_file}" ]]; then

  mkdir -p /srv/salt-overlay/salt/kube-apiserver
  known_tokens_file="/srv/salt-overlay/salt/kube-apiserver/known_tokens.csv"
  (umask u=rw,go= ;
   echo "$KUBELET_TOKEN,kubelet,kubelet" > $known_tokens_file;
   echo "$KUBE_PROXY_TOKEN,kube_proxy,kube_proxy" >> $known_tokens_file)

  mkdir -p /srv/salt-overlay/salt/kubelet
  kubelet_auth_file="/srv/salt-overlay/salt/kubelet/kubernetes_auth"
  (umask u=rw,go= ; echo "{\"BearerToken\": \"$KUBELET_TOKEN\", \"Insecure\": true }" > $kubelet_auth_file)
  kubelet_kubeconfig_file="/srv/salt-overlay/salt/kubelet/kubeconfig"

  mkdir -p /srv/salt-overlay/salt/kubelet
  (umask 077;
  cat > "${kubelet_kubeconfig_file}" << EOF
apiVersion: v1
kind: Config
clusters:
- cluster:
    insecure-skip-tls-verify: true
  name: local
contexts:
- context:
    cluster: local
    user: kubelet
  name: service-account-context
current-context: service-account-context
users:
- name: kubelet
  user:
    token: ${KUBELET_TOKEN}
EOF
)


  mkdir -p /srv/salt-overlay/salt/kube-proxy
  kube_proxy_kubeconfig_file="/srv/salt-overlay/salt/kube-proxy/kubeconfig"
  # Make a kubeconfig file with the token.
  # TODO(etune): put apiserver certs into secret too, and reference from authfile,
  # so that "Insecure" is not needed.
  (umask 077;
  cat > "${kube_proxy_kubeconfig_file}" << EOF
apiVersion: v1
kind: Config
clusters:
- cluster:
    insecure-skip-tls-verify: true
  name: local
contexts:
- context:
    cluster: local
    user: kube-proxy
  name: service-account-context
current-context: service-account-context
users:
- name: kube-proxy
  user:
    token: ${KUBE_PROXY_TOKEN}
EOF
)

  # Generate tokens for other "service accounts".  Append to known_tokens.
  #
  # NB: If this list ever changes, this script actually has to
  # change to detect the existence of this file, kill any deleted
  # old tokens and add any new tokens (to handle the upgrade case).
  service_accounts=("system:scheduler" "system:controller_manager" "system:logging" "system:monitoring" "system:dns")
  for account in "${service_accounts[@]}"; do
    token=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
    echo "${token},${account},${account}" >> "${known_tokens_file}"
  done
fi

readonly BASIC_AUTH_FILE="/srv/salt-overlay/salt/kube-apiserver/basic_auth.csv"
if [[ ! -e "${BASIC_AUTH_FILE}" ]]; then
  mkdir -p /srv/salt-overlay/salt/kube-apiserver
  (umask 077;
    echo "${KUBE_PASSWORD},${KUBE_USER},admin" > "${BASIC_AUTH_FILE}")
fi


# Create the overlay files for the salt tree.  We create these in a separate
# place so that we can blow away the rest of the salt configs on a kube-push and
# re-apply these.

mkdir -p /srv/salt-overlay/pillar
cat <<EOF >/srv/salt-overlay/pillar/cluster-params.sls
instance_prefix: '$(echo "$INSTANCE_PREFIX" | sed -e "s/'/''/g")'
node_instance_prefix: $NODE_INSTANCE_PREFIX
service_cluster_ip_range: $SERVICE_CLUSTER_IP_RANGE
enable_cluster_monitoring: "${ENABLE_CLUSTER_MONITORING:-none}"
enable_cluster_logging: "${ENABLE_CLUSTER_LOGGING:false}"
enable_cluster_ui: "${ENABLE_CLUSTER_UI:true}"
enable_node_logging: "${ENABLE_NODE_LOGGING:false}"
logging_destination: $LOGGING_DESTINATION
elasticsearch_replicas: $ELASTICSEARCH_LOGGING_REPLICAS
enable_cluster_dns: "${ENABLE_CLUSTER_DNS:-false}"
dns_server: $DNS_SERVER_IP
dns_domain: $DNS_DOMAIN
federations_domain_map: ''
e2e_storage_test_environment: "${E2E_STORAGE_TEST_ENVIRONMENT:-false}"
cluster_cidr: "$NODE_IP_RANGES"
allocate_node_cidrs: "${ALLOCATE_NODE_CIDRS:-true}"
admission_control: NamespaceLifecycle,LimitRanger,SecurityContextDeny,ServiceAccount,DefaultStorageClass,ResourceQuota
EOF
