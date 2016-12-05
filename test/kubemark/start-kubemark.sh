#!/bin/bash

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

# Script that creates a Kubemark cluster for a given cloud provider.

TMP_ROOT="$(dirname "${BASH_SOURCE}")/../.."
KUBE_ROOT=$(readlink -e ${TMP_ROOT} 2> /dev/null || perl -MCwd -e 'print Cwd::abs_path shift' ${TMP_ROOT})
KUBECTL="${KUBE_ROOT}/cluster/kubectl.sh"
KUBEMARK_DIRECTORY="${KUBE_ROOT}/test/kubemark"
RESOURCE_DIRECTORY="${KUBEMARK_DIRECTORY}/resources"

source "${KUBE_ROOT}/test/kubemark/skeleton/util.sh"
source "${KUBE_ROOT}/test/kubemark/cloud-provider-config.sh"
source "${KUBE_ROOT}/test/kubemark/${CLOUD_PROVIDER}/util.sh"
source "${KUBE_ROOT}/cluster/kubemark/${CLOUD_PROVIDER}/config-default.sh"

# hack/lib/init.sh will ovewrite ETCD_VERSION if this is unset
# to what is default in hack/lib/etcd.sh
# To avoid it, if it is empty, we set it to 'avoid-overwrite' and
# clean it after that.
if [ -z "${ETCD_VERSION:-}" ]; then
  ETCD_VERSION="avoid-overwrite"
fi
source "${KUBE_ROOT}/hack/lib/init.sh"
if [ "${ETCD_VERSION:-}" == "avoid-overwrite" ]; then
  ETCD_VERSION=""
fi

# Write all environment variables that we need to pass to the kubemark master,
# locally to the file ${RESOURCE_DIRECTORY}/kubemark-master-env.sh.
function create-master-environment-file {
  cat > "${RESOURCE_DIRECTORY}/kubemark-master-env.sh" <<EOF
# Generic variables.
INSTANCE_PREFIX="${INSTANCE_PREFIX:-}"
SERVICE_CLUSTER_IP_RANGE="${SERVICE_CLUSTER_IP_RANGE:-}"

# Etcd related variables.
ETCD_IMAGE="${ETCD_IMAGE:-2.2.1}"
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
STORAGE_BACKEND="${STORAGE_BACKEND:-}"
NUM_NODES="${NUM_NODES:-}"
CUSTOM_ADMISSION_PLUGINS="${CUSTOM_ADMISSION_PLUGINS:-NamespaceLifecycle,LimitRanger,ServiceAccount,ResourceQuota}"
EOF
}

# Generate certs/keys for CA, master, kubelet and kubecfg, and tokens for kubelet
# and kubeproxy.
function generate-pki-config {
  ensure-temp-dir
  gen-kube-bearertoken
  gen-kube-basicauth
  create-certs ${MASTER_IP}
  KUBELET_TOKEN=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
  KUBE_PROXY_TOKEN=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
}

# Write all the relevant certs/keys/tokens to the master.
function write-pki-config-to-master {
  PKI_SETUP_CMD="sudo mkdir /home/kubernetes -p && sudo mkdir /etc/srv/kubernetes -p && \
    sudo bash -c \"echo ${MASTER_CERT_BASE64} | base64 --decode > /etc/srv/kubernetes/server.cert\" && \
    sudo bash -c \"echo ${MASTER_KEY_BASE64} | base64 --decode > /etc/srv/kubernetes/server.key\" && \
    sudo bash -c \"echo ${CA_CERT_BASE64} | base64 --decode > /etc/srv/kubernetes/ca.crt\" && \
    sudo bash -c \"echo ${KUBECFG_CERT_BASE64} | base64 --decode > /etc/srv/kubernetes/kubecfg.crt\" && \
    sudo bash -c \"echo ${KUBECFG_KEY_BASE64} | base64 --decode > /etc/srv/kubernetes/kubecfg.key\" && \
    sudo bash -c \"echo \"${KUBE_BEARER_TOKEN},admin,admin\" > /etc/srv/kubernetes/known_tokens.csv\" && \
    sudo bash -c \"echo \"${KUBELET_TOKEN},kubelet,kubelet\" >> /etc/srv/kubernetes/known_tokens.csv\" && \
    sudo bash -c \"echo \"${KUBE_PROXY_TOKEN},kube_proxy,kube_proxy\" >> /etc/srv/kubernetes/known_tokens.csv\" && \
    sudo bash -c \"echo ${KUBE_PASSWORD},admin,admin > /etc/srv/kubernetes/basic_auth.csv\""
  execute-cmd-on-master "${PKI_SETUP_CMD}"
}

# Copy all the necessary resource files (scripts/configs/manifests) to the master.
function copy-resource-files-to-master {
  copy-files \
  "${SERVER_BINARY_TAR}" \
  "${RESOURCE_DIRECTORY}/kubemark-master-env.sh" \
  "${RESOURCE_DIRECTORY}/start-kubemark-master.sh" \
  "${KUBEMARK_DIRECTORY}/configure-kubectl.sh" \
  "${RESOURCE_DIRECTORY}/manifests/etcd.yaml" \
  "${RESOURCE_DIRECTORY}/manifests/etcd-events.yaml" \
  "${RESOURCE_DIRECTORY}/manifests/kube-apiserver.yaml" \
  "${RESOURCE_DIRECTORY}/manifests/kube-scheduler.yaml" \
  "${RESOURCE_DIRECTORY}/manifests/kube-controller-manager.yaml" \
  "kubernetes@${MASTER_NAME}":/home/kubernetes/
}

# Make startup scripts executable and run start-kubemark-master.sh.
function start-master-components {
  MASTER_STARTUP_CMD="sudo chmod a+x /home/kubernetes/configure-kubectl.sh && \
    sudo chmod a+x /home/kubernetes/start-kubemark-master.sh && \
    sudo bash /home/kubernetes/start-kubemark-master.sh"
  execute-cmd-on-master "${MASTER_STARTUP_CMD}"
}

# Write kubeconfig to ${RESOURCE_DIRECTORY}/kubeconfig.kubemark in order to
# use kubectl locally.
function write-local-kubeconfig {
  LOCAL_KUBECONFIG="${RESOURCE_DIRECTORY}/kubeconfig.kubemark"
  cat > "${LOCAL_KUBECONFIG}" << EOF
apiVersion: v1
kind: Config
users:
- name: admin
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
    user: admin
  name: kubemark-context
current-context: kubemark-context
EOF
}

# Finds the right kubemark binary for 'linux/amd64' platform and uses it to
# create a docker image for hollow-node and upload it to the appropriate
# docker container registry for the cloud provider.
# TODO(shyamjvs): Make the image upload URL and makefile variable w.r.t. provider.
function create-and-upload-hollow-node-image {
  MAKE_DIR="${KUBE_ROOT}/cluster/images/kubemark"
  
  KUBEMARK_BIN="$(kube::util::find-binary-for-platform kubemark linux/amd64)"
  if [[ -z "${KUBEMARK_BIN}" ]]; then
    echo 'Cannot find cmd/kubemark binary'
    exit 1
  fi
  
  echo "Copying kubemark to ${MAKE_DIR}"
  cp "${KUBEMARK_BIN}" "${MAKE_DIR}"
  
  CURR_DIR=`pwd`
  cd "${MAKE_DIR}"
  RETRIES=3
  for attempt in $(seq 1 ${RETRIES}); do
    if ! make; then
      if [[ $((attempt)) -eq "${RETRIES}" ]]; then
        echo "${color_red}Make failed. Exiting.${color_norm}"
        exit 1
      fi
      echo -e "${color_yellow}Make attempt $(($attempt)) failed. Retrying.${color_norm}" >& 2
      sleep $(($attempt * 5))
    else
      break
    fi
  done
  rm kubemark
  cd $CURR_DIR
}

# Generate secret and configMap for the hollow nodes, prepare pod manifests
# for heapster and hollow-node replication controller from templates,
# and finally create these resources through kubectl.
function create-kube-hollow-node-resources {
  KUBECONFIG_CONTENTS=$(echo "apiVersion: v1
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
current-context: kubemark-context" | base64 | tr -d "\n\r")
  
  KUBECONFIG_SECRET="${RESOURCE_DIRECTORY}/kubeconfig_secret.json"
  cat > "${KUBECONFIG_SECRET}" << EOF
{
  "apiVersion": "v1",
  "kind": "Secret",
  "metadata": {
    "name": "kubeconfig"
  },
  "type": "Opaque",
  "data": {
    "kubeconfig": "${KUBECONFIG_CONTENTS}"
  }
}
EOF
  
  NODE_CONFIGMAP="${RESOURCE_DIRECTORY}/node_config_map.json"
  cat > "${NODE_CONFIGMAP}" << EOF
{
  "apiVersion": "v1",
  "kind": "ConfigMap",
  "metadata": {
    "name": "node-configmap"
  },
  "data": {
    "content.type": "${TEST_CLUSTER_API_CONTENT_TYPE}"
  }
}
EOF

  # TODO(shyamjvs): Make path to docker image variable in hollow-node_template.
  sed "s/{{numreplicas}}/${NUM_NODES:-10}/g" "${RESOURCE_DIRECTORY}/hollow-node_template.json" > "${RESOURCE_DIRECTORY}/hollow-node.json"
  sed -i'' -e "s/{{project}}/${PROJECT}/g" "${RESOURCE_DIRECTORY}/hollow-node.json"
  
  mkdir "${RESOURCE_DIRECTORY}/addons" || true
  # TODO(shyamjvs): Make path to docker image variable in heapster_template.
  sed "s/{{MASTER_IP}}/${MASTER_IP}/g" "${RESOURCE_DIRECTORY}/heapster_template.json" > "${RESOURCE_DIRECTORY}/addons/heapster.json"
  metrics_mem_per_node=4
  metrics_mem=$((200 + ${metrics_mem_per_node}*${NUM_NODES:-10}))
  sed -i'' -e "s/{{METRICS_MEM}}/${metrics_mem}/g" "${RESOURCE_DIRECTORY}/addons/heapster.json"
  eventer_mem_per_node=500
  eventer_mem=$((200 * 1024 + ${eventer_mem_per_node}*${NUM_NODES:-10}))
  sed -i'' -e "s/{{EVENTER_MEM}}/${eventer_mem}/g" "${RESOURCE_DIRECTORY}/addons/heapster.json"
  
  "${KUBECTL}" create -f "${RESOURCE_DIRECTORY}/kubemark-ns.json"
  "${KUBECTL}" create -f "${KUBECONFIG_SECRET}" --namespace="kubemark"
  "${KUBECTL}" create -f "${NODE_CONFIGMAP}" --namespace="kubemark"
  "${KUBECTL}" create -f "${RESOURCE_DIRECTORY}/addons" --namespace="kubemark"
  "${KUBECTL}" create configmap node-problem-detector-config --from-file="${RESOURCE_DIRECTORY}/kernel-monitor.json" --namespace="kubemark"
  "${KUBECTL}" create -f "${RESOURCE_DIRECTORY}/hollow-node.json" --namespace="kubemark"
  
  rm "${KUBECONFIG_SECRET}"
  rm "${NODE_CONFIGMAP}"
}

# Wait until all hollow-nodes are running or there is a timeout.
function wait-for-hollow-nodes-to-run-or-timeout {
  echo "Waiting for all HollowNodes to become Running..."
  start=$(date +%s)
  nodes=$("${KUBECTL}" --kubeconfig="${LOCAL_KUBECONFIG}" get node) || true
  ready=$(($(echo "${nodes}" | grep -v "NotReady" | wc -l) - 1))
  
  until [[ "${ready}" -ge "${NUM_NODES}" ]]; do
    echo -n .
    sleep 1
    now=$(date +%s)
    # Fail it if it already took more than 30 minutes.
    if [ $((now - start)) -gt 1800 ]; then
      echo ""
      echo "Timeout waiting for all HollowNodes to become Running"
      # Try listing nodes again - if it fails it means that API server is not responding
      if "${KUBECTL}" --kubeconfig="${LOCAL_KUBECONFIG}" get node &> /dev/null; then
        echo "Found only ${ready} ready Nodes while waiting for ${NUM_NODES}."
      else
        echo "Got error while trying to list Nodes. Probably API server is down."
      fi
      pods=$("${KUBECTL}" get pods --namespace=kubemark) || true
      running=$(($(echo "${pods}" | grep "Running" | wc -l)))
      echo "${running} HollowNode pods are reported as 'Running'"
      not_running=$(($(echo "${pods}" | grep -v "Running" | wc -l) - 1))
      echo "${not_running} HollowNode pods are reported as NOT 'Running'"
      echo $(echo "${pods}" | grep -v "Running")
      exit 1
    fi
    nodes=$("${KUBECTL}" --kubeconfig="${LOCAL_KUBECONFIG}" get node) || true
    ready=$(($(echo "${nodes}" | grep -v "NotReady" | wc -l) - 1))
  done
  echo ""
}

############################### Main Function ########################################
# Setup for master.
find-release-tars
create-master-environment-file
create-master-instance-with-resources
generate-pki-config
wait-for-master-ssh-reachability
write-pki-config-to-master
copy-resource-files-to-master
start-master-components

# Setup for hollow-nodes.
write-local-kubeconfig
create-and-upload-hollow-node-image
create-kube-hollow-node-resources
wait-for-hollow-nodes-to-run-or-timeout

echo "Master IP: ${MASTER_IP}"
echo "Password to kubemark master: ${KUBE_PASSWORD}"
echo "Kubeconfig for kubemark master is written in ${LOCAL_KUBECONFIG}"
