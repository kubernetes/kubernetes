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

# Script that creates a Kubemark cluster with Master running on GCE.

# Hack to make it work for OS X. Ugh...
TMP_ROOT="$(dirname "${BASH_SOURCE}")/../.."
KUBE_ROOT=$(readlink -e ${TMP_ROOT} 2> /dev/null || perl -MCwd -e 'print Cwd::abs_path shift' ${TMP_ROOT})

source "${KUBE_ROOT}/test/kubemark/common.sh"

function writeEnvironmentFile() {
  cat > "${RESOURCE_DIRECTORY}/kubemark-master-env.sh" <<EOF
# Generic variables.
INSTANCE_PREFIX="${INSTANCE_PREFIX:-}"
SERVICE_CLUSTER_IP_RANGE="${SERVICE_CLUSTER_IP_RANGE:-}"

# Etcd related variables.
ETCD_IMAGE="${ETCD_IMAGE:-3.0.14-alpha.1}"
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

writeEnvironmentFile

GCLOUD_COMMON_ARGS="--project ${PROJECT} --zone ${ZONE}"

run-gcloud-compute-with-retries disks create "${MASTER_NAME}-pd" \
  ${GCLOUD_COMMON_ARGS} \
  --type "${MASTER_DISK_TYPE}" \
  --size "${MASTER_DISK_SIZE}"

if [ "${EVENT_PD:-false}" == "true" ]; then
  run-gcloud-compute-with-retries disks create "${MASTER_NAME}-event-pd" \
    ${GCLOUD_COMMON_ARGS} \
    --type "${MASTER_DISK_TYPE}" \
    --size "${MASTER_DISK_SIZE}"
fi

run-gcloud-compute-with-retries addresses create "${MASTER_NAME}-ip" \
  --project "${PROJECT}" \
  --region "${REGION}" -q

MASTER_IP=$(gcloud compute addresses describe "${MASTER_NAME}-ip" \
  --project "${PROJECT}" --region "${REGION}" -q --format='value(address)')

run-gcloud-compute-with-retries instances create "${MASTER_NAME}" \
  ${GCLOUD_COMMON_ARGS} \
  --address "${MASTER_IP}" \
  --machine-type "${MASTER_SIZE}" \
  --image-project="${MASTER_IMAGE_PROJECT}" \
  --image "${MASTER_IMAGE}" \
  --tags "${MASTER_TAG}" \
  --network "${NETWORK}" \
  --scopes "storage-ro,compute-rw,logging-write" \
  --boot-disk-size "${MASTER_ROOT_DISK_SIZE}" \
  --disk "name=${MASTER_NAME}-pd,device-name=master-pd,mode=rw,boot=no,auto-delete=no"

if [ "${EVENT_PD:-false}" == "true" ]; then
  echo "Attaching ${MASTER_NAME}-event-pd to ${MASTER_NAME}"
  run-gcloud-compute-with-retries instances attach-disk "${MASTER_NAME}" \
  ${GCLOUD_COMMON_ARGS} \
  --disk "${MASTER_NAME}-event-pd" \
  --device-name="master-event-pd"
fi

run-gcloud-compute-with-retries firewall-rules create "${INSTANCE_PREFIX}-kubemark-master-https" \
  --project "${PROJECT}" \
  --network "${NETWORK}" \
  --source-ranges "0.0.0.0/0" \
  --target-tags "${MASTER_TAG}" \
  --allow "tcp:443"

ensure-temp-dir
gen-kube-bearertoken
create-certs ${MASTER_IP}
KUBELET_TOKEN=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
KUBE_PROXY_TOKEN=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
HEAPSTER_TOKEN=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
NODE_PROBLEM_DETECTOR_TOKEN=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)

until gcloud compute ssh --zone="${ZONE}" --project="${PROJECT}" "${MASTER_NAME}" --command="ls" &> /dev/null; do
  sleep 1
done

password=$(python -c 'import string,random; print("".join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(16)))')

run-gcloud-compute-with-retries ssh --zone="${ZONE}" --project="${PROJECT}" "${MASTER_NAME}" \
  --command="sudo mkdir /home/kubernetes -p && sudo mkdir /etc/srv/kubernetes -p && \
    sudo bash -c \"echo ${CA_CERT_BASE64} | base64 --decode > /etc/srv/kubernetes/ca.crt\" && \
    sudo bash -c \"echo ${MASTER_CERT_BASE64} | base64 --decode > /etc/srv/kubernetes/server.cert\" && \
    sudo bash -c \"echo ${MASTER_KEY_BASE64} | base64 --decode > /etc/srv/kubernetes/server.key\" && \
    sudo bash -c \"echo ${KUBECFG_CERT_BASE64} | base64 --decode > /etc/srv/kubernetes/kubecfg.crt\" && \
    sudo bash -c \"echo ${KUBECFG_KEY_BASE64} | base64 --decode > /etc/srv/kubernetes/kubecfg.key\" && \
    sudo bash -c \"echo \"${KUBE_BEARER_TOKEN},admin,admin\" > /etc/srv/kubernetes/known_tokens.csv\" && \
    sudo bash -c \"echo \"${KUBELET_TOKEN},system:node:node-name,uid:kubelet,system:nodes\" >> /etc/srv/kubernetes/known_tokens.csv\" && \
    sudo bash -c \"echo \"${KUBE_PROXY_TOKEN},system:kube-proxy,uid:kube_proxy\" >> /etc/srv/kubernetes/known_tokens.csv\" && \
    sudo bash -c \"echo \"${HEAPSTER_TOKEN},system:heapster,uid:heapster\" >> /etc/srv/kubernetes/known_tokens.csv\" && \
    sudo bash -c \"echo \"${NODE_PROBLEM_DETECTOR_TOKEN},system:node-problem-detector,uid:system:node-problem-detector\" >> /etc/srv/kubernetes/known_tokens.csv\" && \
    sudo bash -c \"echo ${password},admin,admin > /etc/srv/kubernetes/basic_auth.csv\""

run-gcloud-compute-with-retries copy-files --zone="${ZONE}" --project="${PROJECT}" \
  "${SERVER_BINARY_TAR}" \
  "${RESOURCE_DIRECTORY}/kubemark-master-env.sh" \
  "${RESOURCE_DIRECTORY}/start-kubemark-master.sh" \
  "${KUBEMARK_DIRECTORY}/configure-kubectl.sh" \
  "${RESOURCE_DIRECTORY}/manifests/etcd.yaml" \
  "${RESOURCE_DIRECTORY}/manifests/etcd-events.yaml" \
  "${RESOURCE_DIRECTORY}/manifests/kube-apiserver.yaml" \
  "${RESOURCE_DIRECTORY}/manifests/kube-scheduler.yaml" \
  "${RESOURCE_DIRECTORY}/manifests/kube-controller-manager.yaml" \
  "${RESOURCE_DIRECTORY}/manifests/kube-addon-manager.yaml" \
  "${RESOURCE_DIRECTORY}/manifests/addons/kubemark-rbac-bindings" \
  "kubernetes@${MASTER_NAME}":/home/kubernetes/

gcloud compute ssh "${MASTER_NAME}" --zone="${ZONE}" --project="${PROJECT}" \
  --command="sudo chmod a+x /home/kubernetes/configure-kubectl.sh && \
    sudo chmod a+x /home/kubernetes/start-kubemark-master.sh && \
    sudo bash /home/kubernetes/start-kubemark-master.sh"

# Setup the docker image for kubemark hollow-node.
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
current-context: kubemark-context")

# Create kubeconfig for Heapster.
HEAPSTER_KUBECONFIG_CONTENTS=$(echo "apiVersion: v1
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
current-context: kubemark-context")

# Create kubeconfig for NodeProblemDetector.
NPD_KUBECONFIG_CONTENTS=$(echo "apiVersion: v1
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
  name: kubemark-npd-context
current-context: kubemark-npd-context")

# Create kubeconfig for local kubectl.
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

sed "s/{{numreplicas}}/${NUM_NODES:-10}/g" "${RESOURCE_DIRECTORY}/hollow-node_template.json" > "${RESOURCE_DIRECTORY}/hollow-node.json"
sed -i'' -e "s/{{project}}/${PROJECT}/g" "${RESOURCE_DIRECTORY}/hollow-node.json"
sed -i'' -e "s/{{master_ip}}/${MASTER_IP}/g" "${RESOURCE_DIRECTORY}/hollow-node.json"

mkdir "${RESOURCE_DIRECTORY}/addons" || true

sed "s/{{MASTER_IP}}/${MASTER_IP}/g" "${RESOURCE_DIRECTORY}/heapster_template.json" > "${RESOURCE_DIRECTORY}/addons/heapster.json"
metrics_mem_per_node=4
metrics_mem=$((200 + ${metrics_mem_per_node}*${NUM_NODES:-10}))
sed -i'' -e "s/{{METRICS_MEM}}/${metrics_mem}/g" "${RESOURCE_DIRECTORY}/addons/heapster.json"
eventer_mem_per_node=500
eventer_mem=$((200 * 1024 + ${eventer_mem_per_node}*${NUM_NODES:-10}))
sed -i'' -e "s/{{EVENTER_MEM}}/${eventer_mem}/g" "${RESOURCE_DIRECTORY}/addons/heapster.json"

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
  --from-literal=npd.kubeconfig="${NPD_KUBECONFIG_CONTENTS}"
# Create addon pods.
"${KUBECTL}" create -f "${RESOURCE_DIRECTORY}/addons" --namespace="kubemark"
# Create the replication controller for hollow-nodes.
"${KUBECTL}" create -f "${RESOURCE_DIRECTORY}/hollow-node.json" --namespace="kubemark"

echo "Waiting for all HollowNodes to become Running..."
start=$(date +%s)
nodes=$("${KUBECTL}" --kubeconfig="${LOCAL_KUBECONFIG}" get node 2> /dev/null) || true
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
  nodes=$("${KUBECTL}" --kubeconfig="${LOCAL_KUBECONFIG}" get node 2> /dev/null) || true
  ready=$(($(echo "${nodes}" | grep -v "NotReady" | wc -l) - 1))
done
echo ""

echo "Master IP: ${MASTER_IP}"
echo "Password to kubemark master: ${password}"
echo "Kubeconfig for kubemark master is written in ${LOCAL_KUBECONFIG}"
