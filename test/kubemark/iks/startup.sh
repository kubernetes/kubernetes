#!/usr/bin/env bash

# Copyright 2018 The Kubernetes Authors.
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

# Script that creates a Kubemark cluster for IBM cloud.

KUBECTL="${KUBE_ROOT}/cluster/kubectl.sh"
KUBEMARK_DIRECTORY="${KUBE_ROOT}/test/kubemark"
RESOURCE_DIRECTORY="${KUBEMARK_DIRECTORY}/resources"

# Generate secret and configMap for the hollow-node pods to work, prepare
# manifests of the hollow-node and heapster replication controllers from
# templates, and finally create these resources through kubectl.
function create-kube-hollow-node-resources {
  # Create kubeconfig for Kubelet.
  KUBELET_KUBECONFIG_CONTENTS=$(echo "apiVersion: v1
kind: Config
users:
- name: kubelet
  user:
    client-certificate-data: "${KUBELET_CERT_BASE64}"
    client-key-data: "${KUBELET_KEY_BASE64}"
clusters:
- name: kubemark
  cluster:
    certificate-authority-data: "${CA_CERT_BASE64}"
    server: https://${MASTER_IP}
contexts:
- context:
    cluster: kubemark
    user: kubelet
  name: kubemark-context
current-context: kubemark-context")

  # Create kubeconfig for Kubeproxy.
  KUBEPROXY_KUBECONFIG_CONTENTS=$(echo "apiVersion: v1
kind: Config
users:
- name: kube-proxy
  user:
    client-certificate-data: "${KUBELET_CERT_BASE64}"
    client-key-data: "${KUBELET_KEY_BASE64}"
clusters:
- name: kubemark
  cluster:
    insecure-skip-tls-verify: true
    server: https://${MASTER_IP}
contexts:
- context:
    cluster: kubemark
    user: kube-proxy
  name: kubemark-context
current-context: kubemark-context")

  # Create kubeconfig for Heapster.
  HEAPSTER_KUBECONFIG_CONTENTS=$(echo "apiVersion: v1
kind: Config
users:
- name: heapster
  user:
    client-certificate-data: "${KUBELET_CERT_BASE64}"
    client-key-data: "${KUBELET_KEY_BASE64}"
clusters:
- name: kubemark
  cluster:
    insecure-skip-tls-verify: true
    server: https://${MASTER_IP}
contexts:
- context:
    cluster: kubemark
    user: heapster
  name: kubemark-context
current-context: kubemark-context")

  # Create kubeconfig for Cluster Autoscaler.
  CLUSTER_AUTOSCALER_KUBECONFIG_CONTENTS=$(echo "apiVersion: v1
kind: Config
users:
- name: cluster-autoscaler
  user:
    client-certificate-data: "${KUBELET_CERT_BASE64}"
    client-key-data: "${KUBELET_KEY_BASE64}"
clusters:
- name: kubemark
  cluster:
    insecure-skip-tls-verify: true
    server: https://${MASTER_IP}
contexts:
- context:
    cluster: kubemark
    user: cluster-autoscaler
  name: kubemark-context
current-context: kubemark-context")

  # Create kubeconfig for NodeProblemDetector.
  NPD_KUBECONFIG_CONTENTS=$(echo "apiVersion: v1
kind: Config
users:
- name: node-problem-detector
  user:
    client-certificate-data: "${KUBELET_CERT_BASE64}"
    client-key-data: "${KUBELET_KEY_BASE64}"
clusters:
- name: kubemark
  cluster:
    insecure-skip-tls-verify: true
    server: https://${MASTER_IP}
contexts:
- context:
    cluster: kubemark
    user: node-problem-detector
  name: kubemark-context
current-context: kubemark-context")

  # Create kubeconfig for Kube DNS.
  KUBE_DNS_KUBECONFIG_CONTENTS=$(echo "apiVersion: v1
kind: Config
users:
- name: kube-dns
  user:
    client-certificate-data: "${KUBELET_CERT_BASE64}"
    client-key-data: "${KUBELET_KEY_BASE64}"
clusters:
- name: kubemark
  cluster:
    insecure-skip-tls-verify: true
    server: https://${MASTER_IP}
contexts:
- context:
    cluster: kubemark
    user: kube-dns
  name: kubemark-context
current-context: kubemark-context")

  # Create kubemark namespace.
  spawn-config
  if kubectl get ns | grep -Fq "kubemark"; then
  	 kubectl delete ns kubemark
  	 while kubectl get ns | grep -Fq "kubemark"
  	 do
  	 	sleep 10
  	 done
  fi
  "${KUBECTL}" create -f "${RESOURCE_DIRECTORY}/kubemark-ns.json"
  # Create configmap for configuring hollow- kubelet, proxy and npd.
  "${KUBECTL}" create configmap "node-configmap" --namespace="kubemark" \
    --from-literal=content.type="${TEST_CLUSTER_API_CONTENT_TYPE}" \
    --from-file=kernel.monitor="${RESOURCE_DIRECTORY}/kernel-monitor.json"

  # Create secret for passing kubeconfigs to kubelet, kubeproxy and npd.
  "${KUBECTL}" create secret generic "kubeconfig" --type=Opaque --namespace="kubemark" \
    --from-literal=kubelet.kubeconfig="${KUBELET_KUBECONFIG_CONTENTS}" \
    --from-literal=kubeproxy.kubeconfig="${KUBEPROXY_KUBECONFIG_CONTENTS}" \
    --from-literal=heapster.kubeconfig="${HEAPSTER_KUBECONFIG_CONTENTS}" \
    --from-literal=cluster_autoscaler.kubeconfig="${CLUSTER_AUTOSCALER_KUBECONFIG_CONTENTS}" \
    --from-literal=npd.kubeconfig="${NPD_KUBECONFIG_CONTENTS}" \
    --from-literal=dns.kubeconfig="${KUBE_DNS_KUBECONFIG_CONTENTS}"

  # Create addon pods.
  # Heapster.
  mkdir -p "${RESOURCE_DIRECTORY}/addons"
  sed "s/{{MASTER_IP}}/${MASTER_IP}/g" "${RESOURCE_DIRECTORY}/heapster_template.json" > "${RESOURCE_DIRECTORY}/addons/heapster.json"
  metrics_mem_per_node=4
  metrics_mem=$((200 + ${metrics_mem_per_node}*${NUM_NODES}))
  sed -i'' -e "s/{{METRICS_MEM}}/${metrics_mem}/g" "${RESOURCE_DIRECTORY}/addons/heapster.json"
  metrics_cpu_per_node_numerator=${NUM_NODES}
  metrics_cpu_per_node_denominator=2
  metrics_cpu=$((80 + metrics_cpu_per_node_numerator / metrics_cpu_per_node_denominator))
  sed -i'' -e "s/{{METRICS_CPU}}/${metrics_cpu}/g" "${RESOURCE_DIRECTORY}/addons/heapster.json"
  eventer_mem_per_node=500
  eventer_mem=$((200 * 1024 + ${eventer_mem_per_node}*${NUM_NODES}))
  sed -i'' -e "s/{{EVENTER_MEM}}/${eventer_mem}/g" "${RESOURCE_DIRECTORY}/addons/heapster.json"

  # Cluster Autoscaler.
  if [[ "${ENABLE_KUBEMARK_CLUSTER_AUTOSCALER:-}" == "true" ]]; then
    echo "Setting up Cluster Autoscaler"
    KUBEMARK_AUTOSCALER_MIG_NAME="${KUBEMARK_AUTOSCALER_MIG_NAME:-${NODE_INSTANCE_PREFIX}-group}"
    KUBEMARK_AUTOSCALER_MIN_NODES="${KUBEMARK_AUTOSCALER_MIN_NODES:-0}"
    KUBEMARK_AUTOSCALER_MAX_NODES="${KUBEMARK_AUTOSCALER_MAX_NODES:-${DESIRED_NODES}}"
    NUM_NODES=${KUBEMARK_AUTOSCALER_MAX_NODES}
    echo "Setting maximum cluster size to ${NUM_NODES}."
    KUBEMARK_MIG_CONFIG="autoscaling.k8s.io/nodegroup: ${KUBEMARK_AUTOSCALER_MIG_NAME}"
    sed "s/{{master_ip}}/${MASTER_IP}/g" "${RESOURCE_DIRECTORY}/cluster-autoscaler_template.json" > "${RESOURCE_DIRECTORY}/addons/cluster-autoscaler.json"
    sed -i'' -e "s/{{kubemark_autoscaler_mig_name}}/${KUBEMARK_AUTOSCALER_MIG_NAME}/g" "${RESOURCE_DIRECTORY}/addons/cluster-autoscaler.json"
    sed -i'' -e "s/{{kubemark_autoscaler_min_nodes}}/${KUBEMARK_AUTOSCALER_MIN_NODES}/g" "${RESOURCE_DIRECTORY}/addons/cluster-autoscaler.json"
    sed -i'' -e "s/{{kubemark_autoscaler_max_nodes}}/${KUBEMARK_AUTOSCALER_MAX_NODES}/g" "${RESOURCE_DIRECTORY}/addons/cluster-autoscaler.json"
  fi

  # Kube DNS.
  if [[ "${ENABLE_KUBEMARK_KUBE_DNS:-}" == "true" ]]; then
    echo "Setting up kube-dns"
    sed "s/{{dns_domain}}/${KUBE_DNS_DOMAIN}/g" "${RESOURCE_DIRECTORY}/kube_dns_template.yaml" > "${RESOURCE_DIRECTORY}/addons/kube_dns.yaml"
  fi

  "${KUBECTL}" create -f "${RESOURCE_DIRECTORY}/addons" --namespace="kubemark"
  set-registry-secrets

  # Create the replication controller for hollow-nodes.
  # We allow to override the NUM_REPLICAS when running Cluster Autoscaler.
  NUM_REPLICAS=${NUM_REPLICAS:-${NUM_NODES}}
  sed "s/{{numreplicas}}/${NUM_REPLICAS}/g" "${RESOURCE_DIRECTORY}/hollow-node_template.yaml" > "${RESOURCE_DIRECTORY}/hollow-node.yaml"
  proxy_cpu=20
  if [ "${NUM_NODES}" -gt 1000 ]; then
    proxy_cpu=50
  fi
  proxy_mem_per_node=50
  proxy_mem=$((100 * 1024 + ${proxy_mem_per_node}*${NUM_NODES}))
  sed -i'' -e "s/{{HOLLOW_PROXY_CPU}}/${proxy_cpu}/g" "${RESOURCE_DIRECTORY}/hollow-node.yaml"
  sed -i'' -e "s/{{HOLLOW_PROXY_MEM}}/${proxy_mem}/g" "${RESOURCE_DIRECTORY}/hollow-node.yaml"
  sed -i'' -e "s'{{kubemark_image_registry}}'${KUBEMARK_IMAGE_REGISTRY}${KUBE_NAMESPACE}'g" "${RESOURCE_DIRECTORY}/hollow-node.yaml"
  sed -i'' -e "s/{{kubemark_image_tag}}/${KUBEMARK_IMAGE_TAG}/g" "${RESOURCE_DIRECTORY}/hollow-node.yaml"
  sed -i'' -e "s/{{master_ip}}/${MASTER_IP}/g" "${RESOURCE_DIRECTORY}/hollow-node.yaml"
  sed -i'' -e "s/{{kubelet_verbosity_level}}/${KUBELET_TEST_LOG_LEVEL}/g" "${RESOURCE_DIRECTORY}/hollow-node.yaml"
  sed -i'' -e "s/{{kubeproxy_verbosity_level}}/${KUBEPROXY_TEST_LOG_LEVEL}/g" "${RESOURCE_DIRECTORY}/hollow-node.yaml"
  sed -i'' -e "s/{{use_real_proxier}}/${USE_REAL_PROXIER}/g" "${RESOURCE_DIRECTORY}/hollow-node.yaml"
  sed -i'' -e "s'{{kubemark_mig_config}}'${KUBEMARK_MIG_CONFIG:-}'g" "${RESOURCE_DIRECTORY}/hollow-node.yaml"
  "${KUBECTL}" create -f "${RESOURCE_DIRECTORY}/hollow-node.yaml" --namespace="kubemark"

  echo "Created secrets, configMaps, replication-controllers required for hollow-nodes."
}

# Wait until all hollow-nodes are running or there is a timeout.
function wait-for-hollow-nodes-to-run-or-timeout {
  echo -n "Waiting for all hollow-nodes to become Running"
  start=$(date +%s)
  nodes=$("${KUBECTL}" --kubeconfig="${KUBECONFIG}" get node 2> /dev/null) || true
  ready=$(($(echo "${nodes}" | grep -v "NotReady" | wc -l) - 1))
  
  until [[ "${ready}" -ge "${NUM_REPLICAS}" ]]; do
    echo -n "."
    sleep 1
    now=$(date +%s)
    # Fail it if it already took more than 30 minutes.
    if [ $((now - start)) -gt 1800 ]; then
      echo ""
      echo -e "${color_red} Timeout waiting for all hollow-nodes to become Running. ${color_norm}"
      # Try listing nodes again - if it fails it means that API server is not responding
      if "${KUBECTL}" --kubeconfig="${KUBECONFIG}" get node &> /dev/null; then
        echo "Found only ${ready} ready hollow-nodes while waiting for ${NUM_NODES}."
      else
        echo "Got error while trying to list hollow-nodes. Probably API server is down."
      fi
      spawn-config
      pods=$("${KUBECTL}" get pods -l name=hollow-node --namespace=kubemark) || true
      running=$(($(echo "${pods}" | grep "Running" | wc -l)))
      echo "${running} hollow-nodes are reported as 'Running'"
      not_running=$(($(echo "${pods}" | grep -v "Running" | wc -l) - 1))
      echo "${not_running} hollow-nodes are reported as NOT 'Running'"
      echo $(echo "${pods}" | grep -v "Running")
      exit 1
    fi
    nodes=$("${KUBECTL}" --kubeconfig="${KUBECONFIG}" get node 2> /dev/null) || true
    ready=$(($(echo "${nodes}" | grep -v "NotReady" | wc -l) - 1))
  done
  echo -e "${color_green} Done!${color_norm}"
}

############################### Main Function ########################################
# In order for the cluster autoscalar to function, the template file must be changed so that the ":443"
# is removed. This is because the port is already given with the MASTER_IP.


# Create clusters and populate with hollow nodes
complete-login
build-kubemark-image
choose-clusters
generate-values
set-hollow-master
echo "Creating kube hollow node resources"
create-kube-hollow-node-resources
master-config
echo -e "${color_blue}EXECUTION COMPLETE${color_norm}"

# Check status of Kubemark
echo -e "${color_yellow}CHECKING STATUS${color_norm}"
wait-for-hollow-nodes-to-run-or-timeout

# Celebrate
echo ""
echo -e "${color_blue}SUCCESS${color_norm}"
clean-repo
exit 0
