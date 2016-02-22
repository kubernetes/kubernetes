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

# deploy the add-on services after the cluster is available

set -e

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
source "${KUBE_ROOT}/cluster/kube-env.sh"
source "${KUBE_ROOT}/cluster/kube-util.sh"

# Get KUBE_MASTER_IP
detect-master

KUBECTL="${KUBE_ROOT}/cluster/kubectl.sh -s ${KUBE_MASTER_IP}:8080"

# Print the ENVs
echo DNS_SERVER_IP=$DNS_SERVER_IP
echo DNS_DOMAIN=$DNS_DOMAIN
echo DNS_REPLICAS=$DNS_REPLICAS
echo KUBE_MASTER_IP=$KUBE_MASTER_IP

function init {
  echo "Creating kube-system namespace..."

  # use kubectl to create kube-system namespace
  NAMESPACE=`eval "${KUBECTL}  get namespaces | grep kube-system | cat"`

  if [ ! "$NAMESPACE" ]; then
    ${KUBECTL} create -f ${KUBE_ROOT}/cluster/docker/kube-config/namespace.yaml 
    echo "The namespace 'kube-system' is successfully created."
  else
    echo "The namespace 'kube-system' is already there. Skipping."
  fi 

  echo
}

function deploy_dns {
  echo "Deploying DNS on Kubernetes"

  sed -e "s/{{ pillar\['dns_replicas'\] }}/${DNS_REPLICAS}/g; \
  s/{{ pillar\['dns_domain'\] }}/${DNS_DOMAIN}/g;s/{kube_server_url}/${KUBE_MASTER_IP}/g;" \
  "${KUBE_ROOT}/cluster/docker/kube-config/skydns-rc.yaml.in" > ./skydns-rc.yaml
  sed -e "s/{{ pillar\['dns_server'\] }}/${DNS_SERVER_IP}/g" \
   "${KUBE_ROOT}/cluster/docker/kube-config/skydns-svc.yaml.in" > ./skydns-svc.yaml

  KUBEDNS=`eval "${KUBECTL} get services --namespace=kube-system | grep kube-dns | cat"`
      
  if [ ! "$KUBEDNS" ]; then
    # use kubectl to create skydns rc and service
    ${KUBECTL} --namespace=kube-system create -f skydns-rc.yaml 
    ${KUBECTL} --namespace=kube-system create -f skydns-svc.yaml

    echo "Kube-dns rc and service is successfully deployed."
  else
    echo "Kube-dns rc and service is already deployed. Skipping."
  fi

  echo
}

# Create kube-system namespace
init

# Deploy DNS rc and pod
deploy_dns
