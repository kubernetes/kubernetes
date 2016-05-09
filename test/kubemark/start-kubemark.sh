#!/bin/bash

# Copyright 2015 The Kubernetes Authors All rights reserved.
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

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..

source "${KUBE_ROOT}/test/kubemark/common.sh"

RUN_FROM_DISTRO=${RUN_FROM_DISTRO:-false}
MAKE_DIR="${KUBE_ROOT}/cluster/images/kubemark"

if [ "${RUN_FROM_DISTRO}" == "false" ]; then
  # Running from repository
  cp "${KUBE_ROOT}/_output/release-stage/server/linux-amd64/kubernetes/server/bin/kubemark" "${MAKE_DIR}"
else
  cp "${KUBE_ROOT}/server/kubernetes-server-linux-amd64.tar.gz" "."
  tar -xzf kubernetes-server-linux-amd64.tar.gz
  cp "kubernetes/server/bin/kubemark" "${MAKE_DIR}"
  rm -rf "kubernetes-server-linux-amd64.tar.gz" "kubernetes"
fi

CURR_DIR=`pwd`
cd "${MAKE_DIR}"
make
rm kubemark
cd $CURR_DIR

GCLOUD_COMMON_ARGS="--project ${PROJECT} --zone ${ZONE}"

run-gcloud-compute-with-retries disks create "${MASTER_NAME}-pd" \
  ${GCLOUD_COMMON_ARGS} \
  --type "${MASTER_DISK_TYPE}" \
  --size "${MASTER_DISK_SIZE}"

run-gcloud-compute-with-retries instances create "${MASTER_NAME}" \
  ${GCLOUD_COMMON_ARGS} \
  --machine-type "${MASTER_SIZE}" \
  --image-project="${MASTER_IMAGE_PROJECT}" \
  --image "${MASTER_IMAGE}" \
  --tags "${MASTER_TAG}" \
  --network "${NETWORK}" \
  --scopes "storage-ro,compute-rw,logging-write" \
  --disk "name=${MASTER_NAME}-pd,device-name=master-pd,mode=rw,boot=no,auto-delete=no"

run-gcloud-compute-with-retries firewall-rules create "${INSTANCE_PREFIX}-kubemark-master-https" \
  --project "${PROJECT}" \
  --network "${NETWORK}" \
  --source-ranges "0.0.0.0/0" \
  --target-tags "${MASTER_TAG}" \
  --allow "tcp:443"

MASTER_IP=$(gcloud compute instances describe ${MASTER_NAME} \
  --zone="${ZONE}" --project="${PROJECT}" | grep natIP: | cut -f2 -d":" | sed "s/ //g")

if [ "${SEPARATE_EVENT_MACHINE:-false}" == "true" ]; then
  EVENT_STORE_NAME="${INSTANCE_PREFIX}-event-store"
    run-gcloud-compute-with-retries disks create "${EVENT_STORE_NAME}-pd" \
      ${GCLOUD_COMMON_ARGS} \
      --type "${MASTER_DISK_TYPE}" \
      --size "${MASTER_DISK_SIZE}"

    run-gcloud-compute-with-retries instances create "${EVENT_STORE_NAME}" \
      ${GCLOUD_COMMON_ARGS} \
      --machine-type "${MASTER_SIZE}" \
      --image-project="${MASTER_IMAGE_PROJECT}" \
      --image "${MASTER_IMAGE}" \
      --tags "${EVENT_STORE_NAME}" \
      --network "${NETWORK}" \
      --scopes "storage-ro,compute-rw,logging-write" \
      --disk "name=${EVENT_STORE_NAME}-pd,device-name=master-pd,mode=rw,boot=no,auto-delete=no"

  EVENT_STORE_IP=$(gcloud compute instances describe ${EVENT_STORE_NAME} \
  --zone="${ZONE}" --project="${PROJECT}" | grep networkIP: | cut -f2 -d":" | sed "s/ //g")

  until gcloud compute ssh --zone="${ZONE}" --project="${PROJECT}" "${EVENT_STORE_NAME}" --command="ls" &> /dev/null; do
    sleep 1
  done

  gcloud compute ssh "${EVENT_STORE_NAME}" --zone="${ZONE}" --project="${PROJECT}" \
    --command="sudo docker run --net=host -d gcr.io/google_containers/etcd:2.0.12 /usr/local/bin/etcd \
      --listen-peer-urls http://127.0.0.1:2380 \
      --addr=127.0.0.1:4002 \
      --bind-addr=0.0.0.0:4002 \
      --data-dir=/var/etcd/data"
fi

ensure-temp-dir
gen-kube-bearertoken
create-certs ${MASTER_IP}
KUBELET_TOKEN=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
KUBE_PROXY_TOKEN=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)

echo "${CA_CERT_BASE64}" | base64 -d > "${RESOURCE_DIRECTORY}/ca.crt"
echo "${KUBECFG_CERT_BASE64}" | base64 -d > "${RESOURCE_DIRECTORY}/kubecfg.crt"
echo "${KUBECFG_KEY_BASE64}" | base64 -d > "${RESOURCE_DIRECTORY}/kubecfg.key"

until gcloud compute ssh --zone="${ZONE}" --project="${PROJECT}" "${MASTER_NAME}" --command="ls" &> /dev/null; do
  sleep 1
done

password=$(python -c 'import string,random; print("".join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(16)))')

gcloud compute ssh --zone="${ZONE}" --project="${PROJECT}" "${MASTER_NAME}" \
  --command="sudo mkdir /srv/kubernetes -p && \
    sudo bash -c \"echo ${MASTER_CERT_BASE64} | base64 -d > /srv/kubernetes/server.cert\" && \
    sudo bash -c \"echo ${MASTER_KEY_BASE64} | base64 -d > /srv/kubernetes/server.key\" && \
    sudo bash -c \"echo ${CA_CERT_BASE64} | base64 -d > /srv/kubernetes/ca.crt\" && \
    sudo bash -c \"echo ${KUBECFG_CERT_BASE64} | base64 -d > /srv/kubernetes/kubecfg.crt\" && \
    sudo bash -c \"echo ${KUBECFG_KEY_BASE64} | base64 -d > /srv/kubernetes/kubecfg.key\" && \
    sudo bash -c \"echo \"${KUBE_BEARER_TOKEN},admin,admin\" > /srv/kubernetes/known_tokens.csv\" && \
    sudo bash -c \"echo \"${KUBELET_TOKEN},kubelet,kubelet\" >> /srv/kubernetes/known_tokens.csv\" && \
    sudo bash -c \"echo \"${KUBE_PROXY_TOKEN},kube_proxy,kube_proxy\" >> /srv/kubernetes/known_tokens.csv\" && \
    sudo bash -c \"echo ${password},admin,admin > /srv/kubernetes/basic_auth.csv\""

if [ "${RUN_FROM_DISTRO}" == "false" ]; then
  gcloud compute copy-files --zone="${ZONE}" --project="${PROJECT}" \
    "${KUBE_ROOT}/_output/release-tars/kubernetes-server-linux-amd64.tar.gz" \
    "${KUBEMARK_DIRECTORY}/start-kubemark-master.sh" \
    "${KUBEMARK_DIRECTORY}/configure-kubectl.sh" \
    "${MASTER_NAME}":~
else
  gcloud compute copy-files --zone="${ZONE}" --project="${PROJECT}" \
    "${KUBE_ROOT}/server/kubernetes-server-linux-amd64.tar.gz" \
    "${KUBEMARK_DIRECTORY}/start-kubemark-master.sh" \
    "${KUBEMARK_DIRECTORY}/configure-kubectl.sh" \
    "${MASTER_NAME}":~
fi

gcloud compute ssh "${MASTER_NAME}" --zone="${ZONE}" --project="${PROJECT}" \
  --command="chmod a+x configure-kubectl.sh && chmod a+x start-kubemark-master.sh && sudo ./start-kubemark-master.sh ${EVENT_STORE_IP:-127.0.0.1}"

# create kubeconfig for Kubelet:
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

LOCAL_KUBECONFIG="${RESOURCE_DIRECTORY}/kubeconfig.loc"
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


sed "s/##numreplicas##/${NUM_NODES:-10}/g" "${RESOURCE_DIRECTORY}/hollow-node_template.json" > "${RESOURCE_DIRECTORY}/hollow-node.json"
sed -i'' -e "s/##project##/${PROJECT}/g" "${RESOURCE_DIRECTORY}/hollow-node.json"

mkdir "${RESOURCE_DIRECTORY}/addons" || true

sed "s/##MASTER_IP##/${MASTER_IP}/g" "${RESOURCE_DIRECTORY}/heapster_template.json" > "${RESOURCE_DIRECTORY}/addons/heapster.json"
metrics_mem_per_node=4
metrics_mem=$((200 + ${metrics_mem_per_node}*${NUM_NODES:-10}))
sed -i'' -e "s/##METRICS_MEM##/${metrics_mem}/g" "${RESOURCE_DIRECTORY}/addons/heapster.json"
eventer_mem_per_node=500
eventer_mem=$((200 * 1024 + ${eventer_mem_per_node}*${NUM_NODES:-10}))
sed -i'' -e "s/##EVENTER_MEM##/${eventer_mem}/g" "${RESOURCE_DIRECTORY}/addons/heapster.json"

"${KUBECTL}" create -f "${RESOURCE_DIRECTORY}/kubemark-ns.json"
"${KUBECTL}" create -f "${KUBECONFIG_SECRET}" --namespace="kubemark"
"${KUBECTL}" create -f "${RESOURCE_DIRECTORY}/addons" --namespace="kubemark"
"${KUBECTL}" create -f "${RESOURCE_DIRECTORY}/hollow-node.json" --namespace="kubemark"

rm "${KUBECONFIG_SECRET}"

echo "Waiting for all HollowNodes to become Running..."
start=$(date +%s)
nodes=$("${KUBECTL}" --kubeconfig="${RESOURCE_DIRECTORY}/kubeconfig.loc" get node) || true
ready=$(($(echo "${nodes}" | grep -v "NotReady" | wc -l) - 1))

until [[ "${ready}" -ge "${NUM_NODES}" ]]; do
  echo -n .
  sleep 1
  now=$(date +%s)
  # Fail it if it already took more than 15 minutes.
  if [ $((now - start)) -gt 900 ]; then
    echo ""
    echo "Timeout waiting for all HollowNodes to become Running"
    # Try listing nodes again - if it fails it means that API server is not responding
    if "${KUBECTL}" --kubeconfig="${RESOURCE_DIRECTORY}/kubeconfig.loc" get node &> /dev/null; then
      echo "Found only ${ready} ready Nodes while waiting for ${NUM_NODES}."
      exit 1
    fi
    echo "Got error while trying to list Nodes. Probably API server is down."
    exit 1
  fi
  nodes=$("${KUBECTL}" --kubeconfig="${RESOURCE_DIRECTORY}/kubeconfig.loc" get node) || true
  ready=$(($(echo "${nodes}" | grep -v "NotReady" | wc -l) - 1))
done
echo ""

echo "Password to kubemark master: ${password}"
