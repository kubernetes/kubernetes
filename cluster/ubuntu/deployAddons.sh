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

# deploy the add-on services after the cluster is available

set -e

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
source "config-default.sh"
KUBECTL="${KUBE_ROOT}/cluster/kubectl.sh"
export KUBECTL_PATH="${KUBE_ROOT}/cluster/ubuntu/binaries/kubectl"
export KUBE_CONFIG_FILE=${KUBE_CONFIG_FILE:-${KUBE_ROOT}/cluster/ubuntu/config-default.sh}

function init {
  echo "Creating kube-system namespace..."
  # use kubectl to create kube-system namespace
  NAMESPACE=`eval "${KUBECTL} get namespaces | grep kube-system | cat"`

  if [ ! "$NAMESPACE" ]; then
    ${KUBECTL} create -f namespace.yaml 
    echo "The namespace 'kube-system' is successfully created."
  else
    echo "The namespace 'kube-system' is already there. Skipping."
  fi 

  echo
}

function deploy_dns {
  echo "Deploying DNS on Kubernetes"
  sed -e "s/\\\$DNS_DOMAIN/${DNS_DOMAIN}/g" "${KUBE_ROOT}/cluster/addons/dns/kubedns-controller.yaml.sed" > kubedns-controller.yaml
  sed -e "s/\\\$DNS_SERVER_IP/${DNS_SERVER_IP}/g" "${KUBE_ROOT}/cluster/addons/dns/kubedns-svc.yaml.sed" > kubedns-svc.yaml

  KUBEDNS=`eval "${KUBECTL} get services --namespace=kube-system | grep kube-dns | cat"`

  if [ ! "$KUBEDNS" ]; then
    # use kubectl to create kubedns controller and service
    ${KUBECTL} --namespace=kube-system create -f kubedns-controller.yaml 
    ${KUBECTL} --namespace=kube-system create -f kubedns-svc.yaml

    echo "Kube-dns controller and service are successfully deployed."
  else
    echo "Kube-dns controller and service are already deployed. Skipping."
  fi

  echo
}

function deploy_dashboard {
    if ${KUBECTL} get deployment -l k8s-app=kubernetes-dashboard --namespace=kube-system | grep kubernetes-dashboard-v &> /dev/null; then
        echo "Kubernetes Dashboard controller already exists"
    else
        echo "Creating Kubernetes Dashboard controller"
        ${KUBECTL} create -f ${KUBE_ROOT}/cluster/addons/dashboard/dashboard-controller.yaml
    fi

    if ${KUBECTL} get service/kubernetes-dashboard --namespace=kube-system &> /dev/null; then
        echo "Kubernetes Dashboard service already exists"
    else
        echo "Creating Kubernetes Dashboard service"
        ${KUBECTL} create -f ${KUBE_ROOT}/cluster/addons/dashboard/dashboard-service.yaml
    fi

  echo
}

init

if [ "${ENABLE_CLUSTER_DNS}" == true ]; then
  deploy_dns
fi

if [ "${ENABLE_CLUSTER_UI}" == true ]; then
  deploy_dashboard
fi

