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

# Perform preparations required to run e2e tests
function prepare-e2e() {
  echo "[DEBUG PREPARE 1] EKS provider doesn't need special preparations for e2e tests" 1>&2
}

# Configure parameters to talk to EKS control plane
function detect-master {
  export KUBE_MASTER_URL="https://82C1C3D0CA24715AA09034CB99C5A5AC.yl4.us-west-2.eks.amazonaws.com"
  if [[ "${KUBE_MASTER_URL}" ]]; then
    echo "[DEBUG PREPARE 2] Using KUBE_MASTER_URL: ${KUBE_MASTER_URL}"
  else
    echo "[DEBUG PREPARE 2] KUBE_MASTER_URL is not defined!"
    # exit 255
  fi
}

echo "[DEBUG PREPARE 3] KUBECTL before:" $(which kubectl)
export KUBECTL="/tmp/aws-k8s-tester/kubectl --kubeconfig=/tmp/aws-k8s-tester/kubeconfig"
echo "[DEBUG PREPARE 4] KUBECTL after:" ${KUBECTL}

export KUBECTL_PATH="/tmp/aws-k8s-tester/kubectl"
echo "[DEBUG PREPARE 5] KUBECTL_PATH:" ${KUBECTL_PATH}

echo "HELLO"
echo "[DEBUG PREPARE 6] env"
env

#export KUBECONFIG="/tmp/aws-k8s-tester/kubeconfig"
#echo "KUBECONFIG:" ${KUBECONFIG}

