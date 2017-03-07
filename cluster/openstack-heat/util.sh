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

# A library of helper functions that each provider hosting Kubernetes must implement to use cluster/kube-*.sh scripts.

# exit on any error
set -e

# Use the config file specified in $KUBE_CONFIG_FILE, or default to
# config-default.sh.
KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
readonly ROOT=$(dirname "${BASH_SOURCE}")
source "${ROOT}/${KUBE_CONFIG_FILE:-"config-default.sh"}"
source "${KUBE_ROOT}/cluster/common.sh"
if [ $CREATE_IMAGE = true ]; then
source "${ROOT}/config-image.sh"
fi

# Verify prereqs on host machine
function verify-prereqs() {
 # Check the OpenStack command-line clients
 for client in swift glance nova heat openstack;
 do
  if which $client >/dev/null 2>&1; then
    echo "${client} client installed"
  else
    echo "${client} client does not exist"
    echo "Please install ${client} client, and retry."
    echo "Documentation for installing ${client} can be found at"
    echo "http://docs.openstack.org/user-guide/common/cli-install-openstack-command-line-clients.html"
    exit 1
  fi
 done
}

# Instantiate a kubernetes cluster
#
# Assumed vars:
#   KUBERNETES_PROVIDER
function kube-up() {
    echo "kube-up for provider ${KUBERNETES_PROVIDER}"
    create-stack
}

# Periodically checks if cluster is created
#
# Assumed vars:
#   STACK_CREATE_TIMEOUT
#   STACK_NAME
function validate-cluster() {

  while (( --$STACK_CREATE_TIMEOUT >= 0)) ;do
     local status=$(openstack stack show "${STACK_NAME}" | awk '$2=="stack_status" {print $4}')
     if [[ $status ]]; then
        echo "Cluster status ${status}"
        if [ $status = "CREATE_COMPLETE" ]; then
          configure-kubectl
          break
        elif [ $status = "CREATE_FAILED" ]; then
          echo "Cluster not created. Please check stack logs to find the problem"
          break
        fi
     else
       echo "Cluster not created. Please verify if process started correctly"
       break
     fi
     sleep 60
  done
}

# Create stack
#
# Assumed vars:
#   OPENSTACK
#   OPENSTACK_TEMP
#   DNS_SERVER
#   OPENSTACK_IP
#   OPENRC_FILE
function create-stack() {
  echo "[INFO] Execute commands to create Kubernetes cluster"
  # It is required for some cloud provider like CityCloud where swift client has different credentials
  source "${ROOT}/openrc-swift.sh"
  upload-resources
  source "${ROOT}/openrc-default.sh"

  create-glance-image

  add-keypair
  run-heat-script
}

# Upload kubernetes release tars and heat templates.
#
# Assumed vars:
#   ROOT
#   KUBERNETES_RELEASE_TAR
function upload-resources() {
  swift post kubernetes --read-acl '.r:*,.rlistings'

  locations=(
    "${ROOT}/../../_output/release-tars/${KUBERNETES_RELEASE_TAR}"
    "${ROOT}/../../server/${KUBERNETES_RELEASE_TAR}"
  )

  RELEASE_TAR_LOCATION=$( (ls -t "${locations[@]}" 2>/dev/null || true) | head -1 )
  RELEASE_TAR_PATH=$(dirname ${RELEASE_TAR_LOCATION})

  echo "[INFO] Uploading ${KUBERNETES_RELEASE_TAR}"
  swift upload kubernetes ${RELEASE_TAR_PATH}/${KUBERNETES_RELEASE_TAR} \
    --object-name kubernetes-server.tar.gz

  echo "[INFO] Uploading kubernetes-salt.tar.gz"
  swift upload kubernetes ${RELEASE_TAR_PATH}/kubernetes-salt.tar.gz \
    --object-name kubernetes-salt.tar.gz
}

# Create a new key pair for use with servers.
#
# Assumed vars:
#   KUBERNETES_KEYPAIR_NAME
#   CLIENT_PUBLIC_KEY_PATH
function add-keypair() {
  local status=$(nova keypair-show ${KUBERNETES_KEYPAIR_NAME})
  if [[ ! $status ]]; then
    nova keypair-add ${KUBERNETES_KEYPAIR_NAME} --pub-key ${CLIENT_PUBLIC_KEY_PATH}
    echo "[INFO] Key pair created"
  else
    echo "[INFO] Key pair already exists"
  fi
}

# Create a new glance image.
#
# Assumed vars:
#   IMAGE_FILE
#   IMAGE_PATH
#   OPENSTACK_IMAGE_NAME
function create-glance-image() {
  if [[ ${CREATE_IMAGE} == "true" ]]; then
    local image_status=$(openstack image show ${OPENSTACK_IMAGE_NAME} | awk '$2=="id" {print $4}')

    if [[ ! $image_status ]]; then
      if [[ "${DOWNLOAD_IMAGE}" == "true" ]]; then
        mkdir -p ${IMAGE_PATH}
        curl -L ${IMAGE_URL_PATH}/${IMAGE_FILE} -o ${IMAGE_PATH}/${IMAGE_FILE} -z ${IMAGE_PATH}/${IMAGE_FILE}
      fi
      echo "[INFO] Create image ${OPENSTACK_IMAGE_NAME}"
      glance image-create --name ${OPENSTACK_IMAGE_NAME} --disk-format ${IMAGE_FORMAT} \
        --container-format ${CONTAINER_FORMAT} --file ${IMAGE_PATH}/${IMAGE_FILE}
    else
      echo "[INFO] Image ${OPENSTACK_IMAGE_NAME} already exists"
    fi
  fi
}

# Create a new kubernetes stack.
#
# Assumed vars:
#   STACK_NAME
#   KUBERNETES_KEYPAIR_NAME
#   DNS_SERVER
#   SWIFT_SERVER_URL
#   OPENSTACK_IMAGE_NAME
#   EXTERNAL_NETWORK
#   IMAGE_ID
#   MASTER_FLAVOR
#   MINION_FLAVOR
#   NUMBER_OF_MINIONS
#   MAX_NUMBER_OF_MINIONS
#   DNS_SERVER
#   STACK_NAME
function run-heat-script() {

  local stack_status=$(openstack stack show ${STACK_NAME})

  # Automatically detect swift url if it wasn't specified
  if [[ -z $SWIFT_SERVER_URL ]]; then
    local rgx=""
    if [ "$OS_IDENTITY_API_VERSION" = "3" ]; then
      rgx="public: (.+)$"
    else
      rgx="publicURL: (.+)$"
    fi
    SWIFT_SERVER_URL=$(openstack catalog show object-store --format value | egrep -o "$rgx" | cut -d" " -f2)
  fi
  local swift_repo_url="${SWIFT_SERVER_URL}/kubernetes"

  if [ $CREATE_IMAGE = true ]; then
    echo "[INFO] Retrieve new image ID"
    IMAGE_ID=$(openstack image show ${OPENSTACK_IMAGE_NAME} | awk '$2=="id" {print $4}')
    echo "[INFO] Image Id ${IMAGE_ID}"
  fi

  if [[ ! $stack_status ]]; then
    echo "[INFO] Create stack ${STACK_NAME}"
    (
      cd ${ROOT}/kubernetes-heat
      openstack stack create --timeout 60 \
      --parameter external_network=${EXTERNAL_NETWORK} \
      --parameter lbaas_version=${LBAAS_VERSION} \
      --parameter fixed_network_cidr=${FIXED_NETWORK_CIDR} \
      --parameter ssh_key_name=${KUBERNETES_KEYPAIR_NAME} \
      --parameter server_image=${IMAGE_ID} \
      --parameter master_flavor=${MASTER_FLAVOR} \
      --parameter minion_flavor=${MINION_FLAVOR} \
      --parameter number_of_minions=${NUMBER_OF_MINIONS} \
      --parameter max_number_of_minions=${MAX_NUMBER_OF_MINIONS} \
      --parameter dns_nameserver=${DNS_SERVER} \
      --parameter kubernetes_salt_url=${swift_repo_url}/kubernetes-salt.tar.gz \
      --parameter kubernetes_server_url=${swift_repo_url}/kubernetes-server.tar.gz \
      --parameter os_auth_url=${OS_AUTH_URL} \
      --parameter os_username=${OS_USERNAME} \
      --parameter os_password=${OS_PASSWORD} \
      --parameter os_region_name=${OS_REGION_NAME} \
      --parameter os_tenant_name=${OS_TENANT_NAME} \
      --parameter os_user_domain_name=${OS_USER_DOMAIN_NAME} \
      --parameter enable_proxy=${ENABLE_PROXY} \
      --parameter ftp_proxy="${FTP_PROXY}" \
      --parameter http_proxy="${HTTP_PROXY}" \
      --parameter https_proxy="${HTTPS_PROXY}" \
      --parameter socks_proxy="${SOCKS_PROXY}" \
      --parameter no_proxy="${NO_PROXY}" \
      --parameter assign_floating_ip="${ASSIGN_FLOATING_IP}" \
      --template kubecluster.yaml \
      ${STACK_NAME}
    )
  else
    echo "[INFO] Stack ${STACK_NAME} already exists"
    openstack stack show ${STACK_NAME}
  fi
}

# Configure kubectl.
#
# Assumed vars:
#   STACK_NAME
function configure-kubectl() {

  export KUBE_MASTER_IP=$(nova show "${STACK_NAME}"-master | awk '$3=="network" {print $6}')
  export CONTEXT="openstack-${STACK_NAME}"
  export KUBE_BEARER_TOKEN="TokenKubelet"

  if [[ "${ENABLE_PROXY:-}" == "true" ]]; then
    echo 'export NO_PROXY=$NO_PROXY,'"${KUBE_MASTER_IP}" > /tmp/kube-proxy-env
    echo 'export no_proxy=$NO_PROXY,'"${KUBE_MASTER_IP}" >> /tmp/kube-proxy-env
    . /tmp/kube-proxy-env
  fi

  create-kubeconfig
}


# Delete a kubernetes cluster
#
# Assumed vars:
#   STACK_NAME
function kube-down {
  source "${ROOT}/openrc-default.sh"
  openstack stack delete ${STACK_NAME}
}

# Perform preparations required to run e2e tests
function prepare-e2e {
  echo "TODO: prepare-e2e" 1>&2
}

function test-build-release {
  echo "test-build-release() " 1>&2
}

# Must ensure that the following ENV vars are set
function detect-master {

  source "${ROOT}/${KUBE_CONFIG_FILE:-"config-default.sh"}"
  source "${ROOT}/openrc-default.sh"

  export KUBE_MASTER_IP=$(nova show "${STACK_NAME}"-master | awk '$3=="network" {print $6}')

  echo "KUBE_MASTER_IP: ${KUBE_MASTER_IP}" 1>&2
}
