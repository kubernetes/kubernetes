#!/usr/bin/env bash

# Copyright 2015 The Kubernetes Authors.
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

# Script that creates a Kubemark cluster for any given cloud provider.

set -o errexit
set -o nounset
set -o pipefail

TMP_ROOT="$(dirname "${BASH_SOURCE}")/../.."
KUBE_ROOT=$(readlink -e "${TMP_ROOT}" 2> /dev/null || perl -MCwd -e 'print Cwd::abs_path shift' "${TMP_ROOT}")

source "${KUBE_ROOT}/test/kubemark/skeleton/util.sh"
source "${KUBE_ROOT}/test/kubemark/cloud-provider-config.sh"
source "${KUBE_ROOT}/test/kubemark/${CLOUD_PROVIDER}/util.sh"
source "${KUBE_ROOT}/cluster/kubemark/${CLOUD_PROVIDER}/config-default.sh"

if [[ -f "${KUBE_ROOT}/test/kubemark/${CLOUD_PROVIDER}/startup.sh" ]] ; then
  source "${KUBE_ROOT}/test/kubemark/${CLOUD_PROVIDER}/startup.sh"
fi

source "${KUBE_ROOT}/cluster/kubemark/util.sh"

# hack/lib/init.sh will ovewrite ETCD_VERSION if this is unset
# what what is default in hack/lib/etcd.sh
# To avoid it, if it is empty, we set it to 'avoid-overwrite' and
# clean it after that.
if [ -z "${ETCD_VERSION:-}" ]; then
  ETCD_VERSION="avoid-overwrite"
fi
source "${KUBE_ROOT}/hack/lib/init.sh"
if [ "${ETCD_VERSION:-}" == "avoid-overwrite" ]; then
  ETCD_VERSION=""
fi

KUBECTL="${KUBE_ROOT}/cluster/kubectl.sh"
KUBEMARK_DIRECTORY="${KUBE_ROOT}/test/kubemark"
RESOURCE_DIRECTORY="${KUBEMARK_DIRECTORY}/resources"

# Generate a random 6-digit alphanumeric tag for the kubemark image.
# Used to uniquify image builds across different invocations of this script.
KUBEMARK_IMAGE_TAG=$(head /dev/urandom | tr -dc 'a-z0-9' | fold -w 6 | head -n 1)

# Write all environment variables that we need to pass to the kubemark master,
# locally to the file ${RESOURCE_DIRECTORY}/kubemark-master-env.sh.
function create-master-environment-file {
  cat > "${RESOURCE_DIRECTORY}/kubemark-master-env.sh" <<EOF
# Generic variables.
INSTANCE_PREFIX="${INSTANCE_PREFIX:-}"
SERVICE_CLUSTER_IP_RANGE="${SERVICE_CLUSTER_IP_RANGE:-}"
EVENT_PD="${EVENT_PD:-}"

# Etcd related variables.
ETCD_IMAGE="${ETCD_IMAGE:-3.2.24-1}"
ETCD_VERSION="${ETCD_VERSION:-}"

# Controller-manager related variables.
CONTROLLER_MANAGER_TEST_ARGS="${CONTROLLER_MANAGER_TEST_ARGS:-}"
ALLOCATE_NODE_CIDRS="${ALLOCATE_NODE_CIDRS:-}"
CLUSTER_IP_RANGE="${CLUSTER_IP_RANGE:-}"
TERMINATED_POD_GC_THRESHOLD="${TERMINATED_POD_GC_THRESHOLD:-}"

# Scheduler related variables.
SCHEDULER_TEST_ARGS="${SCHEDULER_TEST_ARGS:-}"

# Apiserver related variables.
APISERVER_TEST_ARGS="${APISERVER_TEST_ARGS:-}"
STORAGE_MEDIA_TYPE="${STORAGE_MEDIA_TYPE:-}"
STORAGE_BACKEND="${STORAGE_BACKEND:-etcd3}"
ETCD_COMPACTION_INTERVAL_SEC="${ETCD_COMPACTION_INTERVAL_SEC:-}"
RUNTIME_CONFIG="${RUNTIME_CONFIG:-}"
NUM_NODES="${NUM_NODES:-}"
CUSTOM_ADMISSION_PLUGINS="${CUSTOM_ADMISSION_PLUGINS:-}"
FEATURE_GATES="${FEATURE_GATES:-}"
KUBE_APISERVER_REQUEST_TIMEOUT="${KUBE_APISERVER_REQUEST_TIMEOUT:-}"
ENABLE_APISERVER_ADVANCED_AUDIT="${ENABLE_APISERVER_ADVANCED_AUDIT:-}"
EOF
  echo "Created the environment file for master."
}

# Generate certs/keys for CA, master, kubelet and kubecfg, and tokens for kubelet
# and kubeproxy.
function generate-pki-config {
  kube::util::ensure-temp-dir
  gen-kube-bearertoken
  gen-kube-basicauth
  create-certs "${MASTER_IP}"
  KUBELET_TOKEN=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
  KUBE_PROXY_TOKEN=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
  NODE_PROBLEM_DETECTOR_TOKEN=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
  HEAPSTER_TOKEN=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
  CLUSTER_AUTOSCALER_TOKEN=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
  KUBE_DNS_TOKEN=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
  echo "Generated PKI authentication data for kubemark."
}

# Wait for the master to be reachable for executing commands on it. We do this by
# trying to run the bash noop(:) on the master, with 10 retries.
function wait-for-master-reachability {
  execute-cmd-on-master-with-retries ":" 10
  echo "Checked master reachability for remote command execution."
}

# Write all the relevant certs/keys/tokens to the master.
function write-pki-config-to-master {
  PKI_SETUP_CMD="sudo mkdir /home/kubernetes/k8s_auth_data -p && \
    sudo bash -c \"echo ${CA_CERT_BASE64} | base64 --decode > /home/kubernetes/k8s_auth_data/ca.crt\" && \
    sudo bash -c \"echo ${MASTER_CERT_BASE64} | base64 --decode > /home/kubernetes/k8s_auth_data/server.cert\" && \
    sudo bash -c \"echo ${MASTER_KEY_BASE64} | base64 --decode > /home/kubernetes/k8s_auth_data/server.key\" && \
    sudo bash -c \"echo ${REQUESTHEADER_CA_CERT_BASE64} | base64 --decode > /home/kubernetes/k8s_auth_data/aggr_ca.crt\" && \
    sudo bash -c \"echo ${PROXY_CLIENT_CERT_BASE64} | base64 --decode > /home/kubernetes/k8s_auth_data/proxy_client.crt\" && \
    sudo bash -c \"echo ${PROXY_CLIENT_KEY_BASE64} | base64 --decode > /home/kubernetes/k8s_auth_data/proxy_client.key\" && \
    sudo bash -c \"echo ${KUBECFG_CERT_BASE64} | base64 --decode > /home/kubernetes/k8s_auth_data/kubecfg.crt\" && \
    sudo bash -c \"echo ${KUBECFG_KEY_BASE64} | base64 --decode > /home/kubernetes/k8s_auth_data/kubecfg.key\" && \
    sudo bash -c \"echo \"${KUBE_BEARER_TOKEN},admin,admin\" > /home/kubernetes/k8s_auth_data/known_tokens.csv\" && \
    sudo bash -c \"echo \"${KUBELET_TOKEN},system:node:node-name,uid:kubelet,system:nodes\" >> /home/kubernetes/k8s_auth_data/known_tokens.csv\" && \
    sudo bash -c \"echo \"${KUBE_PROXY_TOKEN},system:kube-proxy,uid:kube_proxy\" >> /home/kubernetes/k8s_auth_data/known_tokens.csv\" && \
    sudo bash -c \"echo \"${HEAPSTER_TOKEN},system:heapster,uid:heapster\" >> /home/kubernetes/k8s_auth_data/known_tokens.csv\" && \
    sudo bash -c \"echo \"${CLUSTER_AUTOSCALER_TOKEN},system:cluster-autoscaler,uid:cluster-autoscaler\" >> /home/kubernetes/k8s_auth_data/known_tokens.csv\" && \
    sudo bash -c \"echo \"${NODE_PROBLEM_DETECTOR_TOKEN},system:node-problem-detector,uid:system:node-problem-detector\" >> /home/kubernetes/k8s_auth_data/known_tokens.csv\" && \
    sudo bash -c \"echo \"${KUBE_DNS_TOKEN},system:kube-dns,uid:kube-dns\" >> /home/kubernetes/k8s_auth_data/known_tokens.csv\" && \
    sudo bash -c \"echo ${KUBE_PASSWORD},admin,admin > /home/kubernetes/k8s_auth_data/basic_auth.csv\""
  execute-cmd-on-master-with-retries "${PKI_SETUP_CMD}" 3
  echo "Wrote PKI certs, keys, tokens and admin password to master."
}

# Write kubeconfig to ${RESOURCE_DIRECTORY}/kubeconfig.kubemark in order to
# use kubectl locally.
function write-local-kubeconfig {
  LOCAL_KUBECONFIG="${RESOURCE_DIRECTORY}/kubeconfig.kubemark"
  cat > "${LOCAL_KUBECONFIG}" << EOF
apiVersion: v1
kind: Config
users:
- name: kubecfg
  user:
    client-certificate-data: "${KUBECFG_CERT_BASE64}"
    client-key-data: "${KUBECFG_KEY_BASE64}"
    username: admin
    password: admin
clusters:
- name: kubemark
  cluster:
    certificate-authority-data: "${CA_CERT_BASE64}"
    server: https://${MASTER_IP}
contexts:
- context:
    cluster: kubemark
    user: kubecfg
  name: kubemark-context
current-context: kubemark-context
EOF
  echo "Kubeconfig file for kubemark master written to ${LOCAL_KUBECONFIG}."
}

# Copy all the necessary resource files (scripts/configs/manifests) to the master.
function copy-resource-files-to-master {
  copy-files \
    "${SERVER_BINARY_TAR}" \
    "${RESOURCE_DIRECTORY}/kubemark-master-env.sh" \
    "${RESOURCE_DIRECTORY}/start-kubemark-master.sh" \
    "${RESOURCE_DIRECTORY}/kubeconfig.kubemark" \
    "${KUBEMARK_DIRECTORY}/configure-kubectl.sh" \
    "${RESOURCE_DIRECTORY}/manifests/etcd.yaml" \
    "${RESOURCE_DIRECTORY}/manifests/etcd-events.yaml" \
    "${RESOURCE_DIRECTORY}/manifests/kube-apiserver.yaml" \
    "${RESOURCE_DIRECTORY}/manifests/kube-scheduler.yaml" \
    "${RESOURCE_DIRECTORY}/manifests/kube-controller-manager.yaml" \
    "${RESOURCE_DIRECTORY}/manifests/kube-addon-manager.yaml" \
    "${RESOURCE_DIRECTORY}/manifests/addons/kubemark-rbac-bindings" \
    "kubernetes@${MASTER_NAME}":/home/kubernetes/
  echo "Copied server binary, master startup scripts, configs and resource manifests to master."
}

# Make startup scripts executable and run start-kubemark-master.sh.
function start-master-components {
  echo ""
  MASTER_STARTUP_CMD="sudo bash /home/kubernetes/start-kubemark-master.sh"
  execute-cmd-on-master-with-retries "${MASTER_STARTUP_CMD}"
  echo "The master has started and is now live."
}

# Create a docker image for hollow-node and upload it to the appropriate docker registry.
function create-and-upload-hollow-node-image {
  authenticate-docker
  KUBEMARK_IMAGE_REGISTRY="${KUBEMARK_IMAGE_REGISTRY:-${CONTAINER_REGISTRY}/${PROJECT}}"
  if [[ "${KUBEMARK_BAZEL_BUILD:-}" =~ ^[yY]$ ]]; then
    # Build+push the image through bazel.
    touch WORKSPACE # Needed for bazel.
    build_cmd=("bazel" "run" "//cluster/images/kubemark:push" "--define" "REGISTRY=${KUBEMARK_IMAGE_REGISTRY}" "--define" "IMAGE_TAG=${KUBEMARK_IMAGE_TAG}")
    run-cmd-with-retries "${build_cmd[@]}"
  else
    # Build+push the image through makefile.
    build_cmd=("make" "${KUBEMARK_IMAGE_MAKE_TARGET}")
    MAKE_DIR="${KUBE_ROOT}/cluster/images/kubemark"
    KUBEMARK_BIN="$(kube::util::find-binary-for-platform kubemark linux/amd64)"
    if [[ -z "${KUBEMARK_BIN}" ]]; then
      echo 'Cannot find cmd/kubemark binary'
      exit 1
    fi
    echo "Copying kubemark binary to ${MAKE_DIR}"
    cp "${KUBEMARK_BIN}" "${MAKE_DIR}"
    CURR_DIR=$(pwd)
    cd "${MAKE_DIR}"
    REGISTRY=${KUBEMARK_IMAGE_REGISTRY} IMAGE_TAG=${KUBEMARK_IMAGE_TAG} run-cmd-with-retries "${build_cmd[@]}"
    rm kubemark
    cd "$CURR_DIR"
  fi
  echo "Created and uploaded the kubemark hollow-node image to docker registry."
  # Cleanup the kubemark image after the script exits.
  if [[ "${CLEANUP_KUBEMARK_IMAGE:-}" == "true" ]]; then
    trap delete-kubemark-image EXIT
  fi
}

function delete-kubemark-image {
  delete-image "${KUBEMARK_IMAGE_REGISTRY}/kubemark:${KUBEMARK_IMAGE_TAG}"
}

# Generate secret and configMap for the hollow-node pods to work, prepare
# manifests of the hollow-node and heapster replication controllers from
# templates, and finally create these resources through kubectl.
function create-kube-hollow-node-resources {
  # Create kubeconfig for Kubelet.
  KUBELET_KUBECONFIG_CONTENTS="apiVersion: v1
kind: Config
users:
- name: kubelet
  user:
    client-certificate-data: ${KUBELET_CERT_BASE64}
    client-key-data: ${KUBELET_KEY_BASE64}
clusters:
- name: kubemark
  cluster:
    certificate-authority-data: ${CA_CERT_BASE64}
    server: https://${MASTER_IP}
contexts:
- context:
    cluster: kubemark
    user: kubelet
  name: kubemark-context
current-context: kubemark-context"

  # Create kubeconfig for Kubeproxy.
  KUBEPROXY_KUBECONFIG_CONTENTS="apiVersion: v1
kind: Config
users:
- name: kube-proxy
  user:
    token: ${KUBE_PROXY_TOKEN}
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
current-context: kubemark-context"

  # Create kubeconfig for Heapster.
  HEAPSTER_KUBECONFIG_CONTENTS="apiVersion: v1
kind: Config
users:
- name: heapster
  user:
    token: ${HEAPSTER_TOKEN}
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
current-context: kubemark-context"

  # Create kubeconfig for Cluster Autoscaler.
  CLUSTER_AUTOSCALER_KUBECONFIG_CONTENTS="apiVersion: v1
kind: Config
users:
- name: cluster-autoscaler
  user:
    token: ${CLUSTER_AUTOSCALER_TOKEN}
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
current-context: kubemark-context"

  # Create kubeconfig for NodeProblemDetector.
  NPD_KUBECONFIG_CONTENTS="apiVersion: v1
kind: Config
users:
- name: node-problem-detector
  user:
    token: ${NODE_PROBLEM_DETECTOR_TOKEN}
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
current-context: kubemark-context"

  # Create kubeconfig for Kube DNS.
  KUBE_DNS_KUBECONFIG_CONTENTS="apiVersion: v1
kind: Config
users:
- name: kube-dns
  user:
    token: ${KUBE_DNS_TOKEN}
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
current-context: kubemark-context"

  # Create kubemark namespace.
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
    KUBEMARK_AUTOSCALER_MAX_NODES="${KUBEMARK_AUTOSCALER_MAX_NODES:-10}"
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
  sed -i'' -e "s'{{kubemark_image_registry}}'${KUBEMARK_IMAGE_REGISTRY}'g" "${RESOURCE_DIRECTORY}/hollow-node.yaml"
  sed -i'' -e "s/{{kubemark_image_tag}}/${KUBEMARK_IMAGE_TAG}/g" "${RESOURCE_DIRECTORY}/hollow-node.yaml"
  sed -i'' -e "s/{{master_ip}}/${MASTER_IP}/g" "${RESOURCE_DIRECTORY}/hollow-node.yaml"
  sed -i'' -e "s/{{hollow_kubelet_params}}/${HOLLOW_KUBELET_TEST_ARGS}/g" "${RESOURCE_DIRECTORY}/hollow-node.yaml"
  sed -i'' -e "s/{{hollow_proxy_params}}/${HOLLOW_PROXY_TEST_ARGS}/g" "${RESOURCE_DIRECTORY}/hollow-node.yaml"
  sed -i'' -e "s'{{kubemark_mig_config}}'${KUBEMARK_MIG_CONFIG:-}'g" "${RESOURCE_DIRECTORY}/hollow-node.yaml"
  "${KUBECTL}" create -f "${RESOURCE_DIRECTORY}/hollow-node.yaml" --namespace="kubemark"

  echo "Created secrets, configMaps, replication-controllers required for hollow-nodes."
}

# Wait until all hollow-nodes are running or there is a timeout.
function wait-for-hollow-nodes-to-run-or-timeout {
  echo -n "Waiting for all hollow-nodes to become Running"
  start=$(date +%s)
  nodes=$("${KUBECTL}" --kubeconfig="${LOCAL_KUBECONFIG}" get node 2> /dev/null) || true
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
      if "${KUBECTL}" --kubeconfig="${LOCAL_KUBECONFIG}" get node &> /dev/null; then
        echo "Found only ${ready} ready hollow-nodes while waiting for ${NUM_NODES}."
      else
        echo "Got error while trying to list hollow-nodes. Probably API server is down."
      fi
      pods=$("${KUBECTL}" get pods -l name=hollow-node --namespace=kubemark) || true
      running=$(($(echo "${pods}" | grep "Running" | wc -l)))
      echo "${running} hollow-nodes are reported as 'Running'"
      not_running=$(($(echo "${pods}" | grep -v "Running" | wc -l) - 1))
      echo "${not_running} hollow-nodes are reported as NOT 'Running'"
      echo "${pods}" | grep -v Running
      exit 1
    fi
    nodes=$("${KUBECTL}" --kubeconfig="${LOCAL_KUBECONFIG}" get node 2> /dev/null) || true
    ready=$(($(echo "${nodes}" | grep -v "NotReady" | wc -l) - 1))
  done
  echo -e "${color_green} Done!${color_norm}"
}

############################### Main Function ########################################
detect-project &> /dev/null
find-release-tars

# We need master IP to generate PKI and kubeconfig for cluster.
get-or-create-master-ip
generate-pki-config
write-local-kubeconfig

# Setup for master.
function start-master {
  echo -e "${color_yellow}STARTING SETUP FOR MASTER${color_norm}"
  create-master-environment-file
  create-master-instance-with-resources
  wait-for-master-reachability
  write-pki-config-to-master
  copy-resource-files-to-master
  start-master-components
}
start-master &

# Setup for hollow-nodes.
function start-hollow-nodes {
  echo -e "${color_yellow}STARTING SETUP FOR HOLLOW-NODES${color_norm}"
  create-and-upload-hollow-node-image
  create-kube-hollow-node-resources
  wait-for-hollow-nodes-to-run-or-timeout
}
start-hollow-nodes &

wait
echo ""
echo "Master IP: ${MASTER_IP}"
echo "Password to kubemark master: ${KUBE_PASSWORD}"
echo "Kubeconfig for kubemark master is written in ${LOCAL_KUBECONFIG}"
