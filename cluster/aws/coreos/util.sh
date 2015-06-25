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

# A library of helper functions for CoreOS.

SSH_USER=ubuntu

function detect-minion-image (){
  if [[ -z "${KUBE_MINION_IMAGE-}" ]]; then
    KUBE_MINION_IMAGE=$(curl -s -L http://${COREOS_CHANNEL}.release.core-os.net/amd64-usr/current/coreos_production_ami_all.json | python -c "import json,sys;obj=json.load(sys.stdin);print filter(lambda t: t['name']=='${AWS_REGION}', obj['amis'])[0]['hvm']")
  fi
  if [[ -z "${KUBE_MINION_IMAGE-}" ]]; then
    echo "unable to determine KUBE_MINION_IMAGE"
    exit 2
  fi
}

function generate-minion-user-data() {
  i=$1
  # TODO(bakins): Is this actually used?
  MINION_PRIVATE_IP=$INTERNAL_IP_BASE.1${i}

  # this is a bit of a hack. We make all of our "variables" in
  # our cloud config controlled by env vars from this script
  cat ${KUBE_ROOT}/cluster/aws/coreos/node.yaml
  cat <<EOF
      ENV_TIMESTAMP=$(yaml-quote $(date -u +%Y-%m-%dT%T%z))
      INSTANCE_PREFIX=$(yaml-quote ${INSTANCE_PREFIX})
      SERVER_BINARY_TAR_URL=$(yaml-quote ${SERVER_BINARY_TAR_URL})
      ENABLE_CLUSTER_DNS=$(yaml-quote ${ENABLE_CLUSTER_DNS:-false})
      DNS_SERVER_IP=$(yaml-quote ${DNS_SERVER_IP:-})
      DNS_DOMAIN=$(yaml-quote ${DNS_DOMAIN:-})
      MASTER_IP=$(yaml-quote ${MASTER_INTERNAL_IP})
      MINION_IP=$(yaml-quote ${MINION_PRIVATE_IP})
      KUBELET_TOKEN=$(yaml-quote ${KUBELET_TOKEN:-})
      KUBE_PROXY_TOKEN=$(yaml-quote ${KUBE_PROXY_TOKEN:-})
      KUBE_BEARER_TOKEN=$(yaml-quote ${KUBELET_TOKEN:-})
      KUBERNETES_CONTAINER_RUNTIME=$(yaml-quote ${CONTAINER_RUNTIME})
      RKT_VERSION=$(yaml-quote ${RKT_VERSION})
EOF
}

function check-minion() {
  echo "working"
}

function yaml-quote {
  echo "'$(echo "${@}" | sed -e "s/'/''/g")'"
}
