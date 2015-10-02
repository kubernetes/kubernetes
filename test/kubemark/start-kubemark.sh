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

source "${KUBE_ROOT}/cluster/kubemark/config-default.sh"
source "${KUBE_ROOT}/cluster/kubemark/util.sh"

detect-project &> /dev/null
export PROJECT

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

MASTER_NAME="hollow-cluster-master"

gcloud compute disks create "${MASTER_NAME}-pd" \
    --project "${PROJECT}" \
    --zone "${ZONE}" \
    --type "${MASTER_DISK_TYPE}" \
    --size "${MASTER_DISK_SIZE}"

gcloud compute instances create ${MASTER_NAME} \
    --project "${PROJECT}" \
    --zone "${ZONE}" \
    --machine-type "${MASTER_SIZE}" \
    --image-project="${MASTER_IMAGE_PROJECT}" \
    --image "${MASTER_IMAGE}" \
    --tags "${MASTER_TAG}" \
    --network "${NETWORK}" \
    --scopes "storage-ro,compute-rw,logging-write" \
    --disk "name=${MASTER_NAME}-pd,device-name=master-pd,mode=rw,boot=no,auto-delete=no"

MASTER_IP=$(gcloud compute instances describe hollow-cluster-master \
  --zone="${ZONE}" --project="${PROJECT}" | grep networkIP | cut -f2 -d":" | sed "s/ //g")

until gcloud compute ssh --zone="${ZONE}" --project="${PROJECT}" hollow-cluster-master --command="ls" &> /dev/null; do
  sleep 1
done

if [ "${RUN_FROM_DISTRO}" == "false" ]; then
  gcloud compute copy-files --zone="${ZONE}" --project="${PROJECT}" \
    "${KUBE_ROOT}/_output/release-tars/kubernetes-server-linux-amd64.tar.gz" \
    "${KUBE_ROOT}/test/kubemark/start-kubemark-master.sh" \
    "${KUBE_ROOT}/test/kubemark/configure-kubectl.sh" \
    "hollow-cluster-master":~
else
  gcloud compute copy-files --zone="${ZONE}" --project="${PROJECT}" \
    "${KUBE_ROOT}/server/kubernetes-server-linux-amd64.tar.gz" \
    "${KUBE_ROOT}/test/kubemark/start-kubemark-master.sh" \
    "${KUBE_ROOT}/test/kubemark/configure-kubectl.sh" \
    "hollow-cluster-master":~
fi

gcloud compute ssh --zone=${ZONE} --project="${PROJECT}" hollow-cluster-master \
  --command="chmod a+x configure-kubectl.sh && chmod a+x start-kubemark-master.sh && sudo ./start-kubemark-master.sh"

sed "s/##masterip##/\"${MASTER_IP}\"/g" ${KUBE_ROOT}/test/kubemark/hollow-kubelet_template.json > ${KUBE_ROOT}/test/kubemark/hollow-kubelet.json
sed -i'' -e "s/##numreplicas##/${NUM_MINIONS:-10}/g" ${KUBE_ROOT}/test/kubemark/hollow-kubelet.json
sed -i'' -e "s/##project##/${PROJECT}/g" ${KUBE_ROOT}/test/kubemark/hollow-kubelet.json
kubectl create -f ${KUBE_ROOT}/test/kubemark/kubemark-ns.json
kubectl create -f ${KUBE_ROOT}/test/kubemark/hollow-kubelet.json --namespace="kubemark"
