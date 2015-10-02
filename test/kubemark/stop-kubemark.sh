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

# Script that destroys Kubemark clusters and deletes all GCE resources created for Master
KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..

source "${KUBE_ROOT}/cluster/kubemark/config-default.sh"
source "${KUBE_ROOT}/cluster/kubemark/util.sh"

detect-project &> /dev/null

MASTER_NAME="hollow-cluster-master"

kubectl delete -f ${KUBE_ROOT}/test/kubemark/hollow-kubelet.json &> /dev/null || true
kubectl delete -f ${KUBE_ROOT}/test/kubemark/kubemark-ns.json &> /dev/null || true

gcloud compute instances delete "${MASTER_NAME}" \
    --project "${PROJECT}" \
    --quiet \
    --zone "${ZONE}" || true

gcloud compute disks delete \
      --project "${PROJECT}" \
      --quiet \
      --zone "${ZONE}" \
      "${MASTER_NAME}"-pd || true

rm -rf "${KUBE_ROOT}/test/kubemark/kubeconfig.loc" &> /dev/null || true
