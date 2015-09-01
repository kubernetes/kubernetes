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

gcloud compute disks create "${MASTER_NAME}-pd" \
    ${GCLOUD_COMMON_ARGS} \
    --type "${MASTER_DISK_TYPE}" \
    --size "${MASTER_DISK_SIZE}"

gcloud compute instances create "${MASTER_NAME}" \
    ${GCLOUD_COMMON_ARGS} \
    --machine-type "${MASTER_SIZE}" \
    --image-project="${MASTER_IMAGE_PROJECT}" \
    --image "${MASTER_IMAGE}" \
    --tags "${MASTER_TAG}" \
    --network "${NETWORK}" \
    --scopes "storage-ro,compute-rw,logging-write" \
    --disk "name=${MASTER_NAME}-pd,device-name=master-pd,mode=rw,boot=no,auto-delete=no"

gcloud compute firewall-rules create "${INSTANCE_PREFIX}-kubemark-master-https" \
    --project "${PROJECT}" \
    --network "${NETWORK}" \
    --source-ranges "0.0.0.0/0" \
    --target-tags "${MASTER_TAG}" \
    --allow "tcp:443"

MASTER_IP=$(gcloud compute instances describe ${MASTER_NAME} \
  --zone="${ZONE}" --project="${PROJECT}" | grep natIP: | cut -f2 -d":" | sed "s/ //g")

if [ "${SEPARATE_EVENT_MACHINE:-false}" == "true" ]; then
  EVENT_STORE_NAME="${INSTANCE_PREFIX}-event-store"
  gcloud compute disks create "${EVENT_STORE_NAME}-pd" \
      ${GCLOUD_COMMON_ARGS} \
      --type "${MASTER_DISK_TYPE}" \
      --size "${MASTER_DISK_SIZE}"

  gcloud compute instances create "${EVENT_STORE_NAME}" \
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

  gcloud compute ssh ${EVENT_STORE_NAME} --zone=${ZONE} --project="${PROJECT}" \
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

echo "${CA_CERT_BASE64}" | base64 -d > ca.crt
echo "${KUBECFG_CERT_BASE64}" | base64 -d > kubecfg.crt
echo "${KUBECFG_KEY_BASE64}" | base64 -d > kubecfg.key

until gcloud compute ssh --zone="${ZONE}" --project="${PROJECT}" "${MASTER_NAME}" --command="ls" &> /dev/null; do
  sleep 1
done

gcloud compute ssh --zone=${ZONE} --project="${PROJECT}" ${MASTER_NAME} \
  --command="sudo mkdir /srv/kubernetes -p && \
  sudo bash -c \"echo ${MASTER_CERT_BASE64} | base64 -d > /srv/kubernetes/server.cert\" && \
  sudo bash -c \"echo ${MASTER_KEY_BASE64} | base64 -d > /srv/kubernetes/server.key\" && \
  sudo bash -c \"echo ${CA_CERT_BASE64} | base64 -d > /srv/kubernetes/ca.crt\" && \
  sudo bash -c \"echo ${KUBECFG_CERT_BASE64} | base64 -d > /srv/kubernetes/kubecfg.crt\" && \
  sudo bash -c \"echo ${KUBECFG_KEY_BASE64} | base64 -d > /srv/kubernetes/kubecfg.key\" && \
  sudo bash -c \"echo \"${KUBE_BEARER_TOKEN},admin,admin\" > /srv/kubernetes/known_tokens.csv\" && \
  sudo bash -c \"echo \"${KUBELET_TOKEN},kubelet,kubelet\" >> /srv/kubernetes/known_tokens.csv\" && \
  sudo bash -c \"echo \"${KUBE_PROXY_TOKEN},kube_proxy,kube_proxy\" >> /srv/kubernetes/known_tokens.csv\" && \
  sudo bash -c \"echo admin,admin,admin > /srv/kubernetes/basic_auth.csv\""

if [ "${RUN_FROM_DISTRO}" == "false" ]; then
  gcloud compute copy-files --zone="${ZONE}" --project="${PROJECT}" \
    "${KUBE_ROOT}/_output/release-tars/kubernetes-server-linux-amd64.tar.gz" \
    "${KUBE_ROOT}/test/kubemark/start-kubemark-master.sh" \
    "${KUBE_ROOT}/test/kubemark/configure-kubectl.sh" \
    "${MASTER_NAME}":~
else
  gcloud compute copy-files --zone="${ZONE}" --project="${PROJECT}" \
    "${KUBE_ROOT}/server/kubernetes-server-linux-amd64.tar.gz" \
    "${KUBE_ROOT}/test/kubemark/start-kubemark-master.sh" \
    "${KUBE_ROOT}/test/kubemark/configure-kubectl.sh" \
    "${MASTER_NAME}":~
fi

gcloud compute ssh ${MASTER_NAME} --zone=${ZONE} --project="${PROJECT}" \
  --command="chmod a+x configure-kubectl.sh && chmod a+x start-kubemark-master.sh && sudo ./start-kubemark-master.sh ${EVENT_STORE_IP:-127.0.0.1}"

# create kubeconfig for Kubelet:
KUBECONFIG_CONTENTS=$(echo "apiVersion: v1
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
current-context: kubemark-context" | base64 | tr -d "\n\r")

KUBECONFIG_SECRET=kubeconfig_secret.json
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

LOCAL_KUBECONFIG=${KUBE_ROOT}/test/kubemark/kubeconfig.loc
cat > "${LOCAL_KUBECONFIG}" << EOF
apiVersion: v1
kind: Config
users:
- name: admin
  user:
    client-certificate-data: ${KUBECFG_CERT_BASE64}
    client-key-data: ${KUBECFG_KEY_BASE64}
    username: admin
    password: admin
clusters:
- name: kubemark
  cluster:
    certificate-authority-data: ${CA_CERT_BASE64}
    server: https://${MASTER_IP}
contexts:
- context:
    cluster: kubemark
    user: admin
  name: kubemark-context
current-context: kubemark-context
EOF

sed "s/##numreplicas##/${NUM_MINIONS:-10}/g" ${KUBE_ROOT}/test/kubemark/hollow-node_template.json > ${KUBE_ROOT}/test/kubemark/hollow-node.json
sed -i'' -e "s/##project##/${PROJECT}/g" ${KUBE_ROOT}/test/kubemark/hollow-node.json
kubectl create -f ${KUBE_ROOT}/test/kubemark/kubemark-ns.json
kubectl create -f ${KUBECONFIG_SECRET} --namespace="kubemark"
kubectl create -f ${KUBE_ROOT}/test/kubemark/hollow-node.json --namespace="kubemark"

rm ${KUBECONFIG_SECRET}

echo "Waiting for all HollowNodes to become Running..."
echo "This can loop forever if something crashed."
until [[ "$(kubectl --kubeconfig=${KUBE_ROOT}/test/kubemark/kubeconfig.loc get node | grep Ready | wc -l)" == "${NUM_MINIONS}" ]]; do
  echo -n .
  sleep 1
done
echo ""
