#!/bin/bash

# Copyright 2014 The Kubernetes Authors.
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

# A library of helper functions for deploying on Rackspace

# Use the config file specified in $KUBE_CONFIG_FILE, or default to
# config-default.sh.
KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
source $(dirname ${BASH_SOURCE})/${KUBE_CONFIG_FILE-"config-default.sh"}
source "${KUBE_ROOT}/cluster/common.sh"
source "${KUBE_ROOT}/cluster/rackspace/authorization.sh"

verify-prereqs() {
  # Make sure that prerequisites are installed.
  for x in nova swiftly; do
    if [ "$(which $x)" == "" ]; then
      echo "cluster/rackspace/util.sh:  Can't find $x in PATH, please fix and retry."
      exit 1
    fi
  done

  if [[ -z "${OS_AUTH_URL-}" ]]; then
    echo "cluster/rackspace/util.sh: OS_AUTH_URL not set."
    echo -e "\texport OS_AUTH_URL=https://identity.api.rackspacecloud.com/v2.0/"
    return 1
  fi

  if [[ -z "${OS_USERNAME-}" ]]; then
    echo "cluster/rackspace/util.sh: OS_USERNAME not set."
    echo -e "\texport OS_USERNAME=myusername"
    return 1
  fi

  if [[ -z "${OS_PASSWORD-}" ]]; then
    echo "cluster/rackspace/util.sh: OS_PASSWORD not set."
    echo -e "\texport OS_PASSWORD=myapikey"
    return 1
  fi
}

rax-ssh-key() {
  if [ ! -f $HOME/.ssh/${SSH_KEY_NAME} ]; then
    echo "cluster/rackspace/util.sh: Generating SSH KEY ${HOME}/.ssh/${SSH_KEY_NAME}"
    ssh-keygen -f ${HOME}/.ssh/${SSH_KEY_NAME} -N '' > /dev/null
  fi

  if ! $(nova keypair-list | grep $SSH_KEY_NAME > /dev/null 2>&1); then
    echo "cluster/rackspace/util.sh: Uploading key to Rackspace:"
    echo -e "\tnova keypair-add ${SSH_KEY_NAME} --pub-key ${HOME}/.ssh/${SSH_KEY_NAME}.pub"
    nova keypair-add ${SSH_KEY_NAME} --pub-key ${HOME}/.ssh/${SSH_KEY_NAME}.pub > /dev/null 2>&1
  else
    echo "cluster/rackspace/util.sh: SSH key ${SSH_KEY_NAME}.pub already uploaded"
  fi
}

rackspace-set-vars() {

  CLOUDFILES_CONTAINER="kubernetes-releases-${OS_USERNAME}"
  CONTAINER_PREFIX=${CONTAINER_PREFIX-devel/}
  find-release-tars
}

# Retrieves a tempurl from cloudfiles to make the release object publicly accessible temporarily.
find-object-url() {

  rackspace-set-vars

  KUBE_TAR=${CLOUDFILES_CONTAINER}/${CONTAINER_PREFIX}/kubernetes-server-linux-amd64.tar.gz

 # Create temp URL good for 24 hours
  RELEASE_TMP_URL=$(swiftly -A ${OS_AUTH_URL} -U ${OS_USERNAME} -K ${OS_PASSWORD} tempurl GET ${KUBE_TAR} 86400 )
  echo "cluster/rackspace/util.sh: Object temp URL:"
  echo -e "\t${RELEASE_TMP_URL}"

}

ensure_dev_container() {

  SWIFTLY_CMD="swiftly -A ${OS_AUTH_URL} -U ${OS_USERNAME} -K ${OS_PASSWORD}"

  if ! ${SWIFTLY_CMD} get ${CLOUDFILES_CONTAINER} > /dev/null 2>&1 ; then
    echo "cluster/rackspace/util.sh: Container doesn't exist. Creating container ${CLOUDFILES_CONTAINER}"
    ${SWIFTLY_CMD} put ${CLOUDFILES_CONTAINER} > /dev/null 2>&1
  fi
}

# Copy kubernetes-server-linux-amd64.tar.gz to cloud files object store
copy_dev_tarballs() {

  echo "cluster/rackspace/util.sh: Uploading to Cloud Files"
  ${SWIFTLY_CMD} put -i ${SERVER_BINARY_TAR} \
  ${CLOUDFILES_CONTAINER}/${CONTAINER_PREFIX}/kubernetes-server-linux-amd64.tar.gz > /dev/null 2>&1

  echo "Release pushed."
}

prep_known_tokens() {
  for (( i=0; i<${#NODE_NAMES[@]}; i++)); do
    generate_kubelet_tokens ${NODE_NAMES[i]}
    cat ${KUBE_TEMP}/${NODE_NAMES[i]}_tokens.csv >> ${KUBE_TEMP}/known_tokens.csv
  done

    # Generate tokens for other "service accounts".  Append to known_tokens.
    #
    # NB: If this list ever changes, this script actually has to
    # change to detect the existence of this file, kill any deleted
    # old tokens and add any new tokens (to handle the upgrade case).
    local -r service_accounts=("system:scheduler" "system:controller_manager" "system:logging" "system:monitoring" "system:dns")
    for account in "${service_accounts[@]}"; do
      echo "$(create_token),${account},${account}" >> ${KUBE_TEMP}/known_tokens.csv
    done

  generate_admin_token
}

rax-boot-master() {

  DISCOVERY_URL=$(curl https://discovery.etcd.io/new?size=1)
  DISCOVERY_ID=$(echo "${DISCOVERY_URL}" | cut -f 4 -d /)
  echo "cluster/rackspace/util.sh: etcd discovery URL: ${DISCOVERY_URL}"

# Copy cloud-config to KUBE_TEMP and work some sed magic
  sed -e "s|DISCOVERY_ID|${DISCOVERY_ID}|" \
      -e "s|CLOUD_FILES_URL|${RELEASE_TMP_URL//&/\\&}|" \
      -e "s|KUBE_USER|${KUBE_USER}|" \
      -e "s|KUBE_PASSWORD|${KUBE_PASSWORD}|" \
      -e "s|SERVICE_CLUSTER_IP_RANGE|${SERVICE_CLUSTER_IP_RANGE}|" \
      -e "s|KUBE_NETWORK|${KUBE_NETWORK}|" \
      -e "s|OS_AUTH_URL|${OS_AUTH_URL}|" \
      -e "s|OS_USERNAME|${OS_USERNAME}|" \
      -e "s|OS_PASSWORD|${OS_PASSWORD}|" \
      -e "s|OS_TENANT_NAME|${OS_TENANT_NAME}|" \
      -e "s|OS_REGION_NAME|${OS_REGION_NAME}|" \
      $(dirname $0)/rackspace/cloud-config/master-cloud-config.yaml > $KUBE_TEMP/master-cloud-config.yaml


  MASTER_BOOT_CMD="nova boot \
--key-name ${SSH_KEY_NAME} \
--flavor ${KUBE_MASTER_FLAVOR} \
--image ${KUBE_IMAGE} \
--meta ${MASTER_TAG} \
--meta ETCD=${DISCOVERY_ID} \
--user-data ${KUBE_TEMP}/master-cloud-config.yaml \
--config-drive true \
--nic net-id=${NETWORK_UUID} \
${MASTER_NAME}"

  echo "cluster/rackspace/util.sh: Booting ${MASTER_NAME} with following command:"
  echo -e "\t$MASTER_BOOT_CMD"
  $MASTER_BOOT_CMD
}

rax-boot-nodes() {

  cp $(dirname $0)/rackspace/cloud-config/node-cloud-config.yaml \
  ${KUBE_TEMP}/node-cloud-config.yaml

  for (( i=0; i<${#NODE_NAMES[@]}; i++)); do

    get_tokens_from_csv ${NODE_NAMES[i]}

    sed -e "s|DISCOVERY_ID|${DISCOVERY_ID}|" \
        -e "s|CLOUD_FILES_URL|${RELEASE_TMP_URL//&/\\&}|" \
        -e "s|DNS_SERVER_IP|${DNS_SERVER_IP:-}|" \
        -e "s|DNS_DOMAIN|${DNS_DOMAIN:-}|" \
        -e "s|ENABLE_CLUSTER_DNS|${ENABLE_CLUSTER_DNS:-false}|" \
        -e "s|ENABLE_NODE_LOGGING|${ENABLE_NODE_LOGGING:-false}|" \
        -e "s|INDEX|$((i + 1))|g" \
        -e "s|KUBELET_TOKEN|${KUBELET_TOKEN}|" \
        -e "s|KUBE_NETWORK|${KUBE_NETWORK}|" \
        -e "s|KUBELET_TOKEN|${KUBELET_TOKEN}|" \
        -e "s|KUBE_PROXY_TOKEN|${KUBE_PROXY_TOKEN}|" \
        -e "s|LOGGING_DESTINATION|${LOGGING_DESTINATION:-}|" \
    $(dirname $0)/rackspace/cloud-config/node-cloud-config.yaml > $KUBE_TEMP/node-cloud-config-$(($i + 1)).yaml


    NODE_BOOT_CMD="nova boot \
--key-name ${SSH_KEY_NAME} \
--flavor ${KUBE_NODE_FLAVOR} \
--image ${KUBE_IMAGE} \
--meta ${NODE_TAG} \
--user-data ${KUBE_TEMP}/node-cloud-config-$(( i +1 )).yaml \
--config-drive true \
--nic net-id=${NETWORK_UUID} \
${NODE_NAMES[$i]}"

    echo "cluster/rackspace/util.sh: Booting ${NODE_NAMES[$i]} with following command:"
    echo -e "\t$NODE_BOOT_CMD"
    $NODE_BOOT_CMD
  done
}

rax-nova-network() {
  if ! $(nova network-list | grep $NOVA_NETWORK_LABEL > /dev/null 2>&1); then
    SAFE_CIDR=$(echo $NOVA_NETWORK_CIDR | tr -d '\\')
    NETWORK_CREATE_CMD="nova network-create $NOVA_NETWORK_LABEL $SAFE_CIDR"

    echo "cluster/rackspace/util.sh: Creating cloud network with following command:"
    echo -e "\t${NETWORK_CREATE_CMD}"

    $NETWORK_CREATE_CMD
  else
    echo "cluster/rackspace/util.sh: Using existing cloud network $NOVA_NETWORK_LABEL"
  fi
}

detect-nodes() {
  KUBE_NODE_IP_ADDRESSES=()
  for (( i=0; i<${#NODE_NAMES[@]}; i++)); do
    local node_ip=$(nova show --minimal ${NODE_NAMES[$i]} \
      | grep accessIPv4 | awk '{print $4}')
    echo "cluster/rackspace/util.sh: Found ${NODE_NAMES[$i]} at ${node_ip}"
    KUBE_NODE_IP_ADDRESSES+=("${node_ip}")
  done
  if [ -z "$KUBE_NODE_IP_ADDRESSES" ]; then
    echo "cluster/rackspace/util.sh: Could not detect Kubernetes node nodes.  Make sure you've launched a cluster with 'kube-up.sh'"
    exit 1
  fi

}

detect-master() {
  KUBE_MASTER=${MASTER_NAME}

  echo "Waiting for ${MASTER_NAME} IP Address."
  echo
  echo "  This will continually check to see if the master node has an IP address."
  echo

  KUBE_MASTER_IP=$(nova show $KUBE_MASTER --minimal | grep accessIPv4 | awk '{print $4}')

  while [ "${KUBE_MASTER_IP-|}" == "|" ]; do
    KUBE_MASTER_IP=$(nova show $KUBE_MASTER --minimal | grep accessIPv4 | awk '{print $4}')
    printf "."
    sleep 2
  done

  echo "${KUBE_MASTER} IP Address is ${KUBE_MASTER_IP}"
}

# $1 should be the network you would like to get an IP address for
detect-master-nova-net() {
  KUBE_MASTER=${MASTER_NAME}

  MASTER_IP=$(nova show $KUBE_MASTER --minimal | grep $1 | awk '{print $5}')
}

kube-up() {

  SCRIPT_DIR=$(CDPATH="" cd $(dirname $0); pwd)

  rackspace-set-vars
  ensure_dev_container
  copy_dev_tarballs

  # Find the release to use.  Generally it will be passed when doing a 'prod'
  # install and will default to the release/config.sh version when doing a
  # developer up.
  find-object-url

  # Create a temp directory to hold scripts that will be uploaded to master/nodes
  KUBE_TEMP=$(mktemp -d -t kubernetes.XXXXXX)
  trap "rm -rf ${KUBE_TEMP}" EXIT

  load-or-gen-kube-basicauth
  python2.7 $(dirname $0)/../third_party/htpasswd/htpasswd.py -b -c ${KUBE_TEMP}/htpasswd $KUBE_USER $KUBE_PASSWORD
  HTPASSWD=$(cat ${KUBE_TEMP}/htpasswd)

  rax-nova-network
  NETWORK_UUID=$(nova network-list | grep -i ${NOVA_NETWORK_LABEL} | awk '{print $2}')

  # create and upload ssh key if necessary
  rax-ssh-key

  echo "cluster/rackspace/util.sh: Starting Cloud Servers"
  prep_known_tokens

  rax-boot-master
  rax-boot-nodes

  detect-master

  # TODO look for a better way to get the known_tokens to the master. This is needed over file injection since the files were too large on a 4 node cluster.
  $(scp -o StrictHostKeyChecking=no -i ~/.ssh/${SSH_KEY_NAME} ${KUBE_TEMP}/known_tokens.csv core@${KUBE_MASTER_IP}:/home/core/known_tokens.csv)
  $(sleep 2)
  $(ssh -o StrictHostKeyChecking=no -i ~/.ssh/${SSH_KEY_NAME} core@${KUBE_MASTER_IP} sudo /usr/bin/mkdir -p /var/lib/kube-apiserver)
  $(ssh -o StrictHostKeyChecking=no -i ~/.ssh/${SSH_KEY_NAME} core@${KUBE_MASTER_IP} sudo mv /home/core/known_tokens.csv /var/lib/kube-apiserver/known_tokens.csv)
  $(ssh -o StrictHostKeyChecking=no -i ~/.ssh/${SSH_KEY_NAME} core@${KUBE_MASTER_IP} sudo chown root.root /var/lib/kube-apiserver/known_tokens.csv)
  $(ssh -o StrictHostKeyChecking=no -i ~/.ssh/${SSH_KEY_NAME} core@${KUBE_MASTER_IP} sudo systemctl restart kube-apiserver)

  FAIL=0
  for job in `jobs -p`
  do
      wait $job || let "FAIL+=1"
  done
  if (( $FAIL != 0 )); then
    echo "${FAIL} commands failed.  Exiting."
    exit 2
  fi

  echo "Waiting for cluster initialization."
  echo
  echo "  This will continually check to see if the API for kubernetes is reachable."
  echo "  This might loop forever if there was some uncaught error during start"
  echo "  up."
  echo

  #This will fail until apiserver salt is updated
  until $(curl --insecure --user ${KUBE_USER}:${KUBE_PASSWORD} --max-time 5 \
          --fail --output /dev/null --silent https://${KUBE_MASTER_IP}/healthz); do
      printf "."
      sleep 2
  done

  echo "Kubernetes cluster created."

  export KUBE_CERT=""
  export KUBE_KEY=""
  export CA_CERT=""
  export CONTEXT="rackspace_${INSTANCE_PREFIX}"

  create-kubeconfig

  # Don't bail on errors, we want to be able to print some info.
  set +e

  detect-nodes

  # ensures KUBECONFIG is set
  get-kubeconfig-basicauth
  echo "All nodes may not be online yet, this is okay."
  echo
  echo "Kubernetes cluster is running.  The master is running at:"
  echo
  echo "  https://${KUBE_MASTER_IP}"
  echo
  echo "The user name and password to use is located in ${KUBECONFIG:-$DEFAULT_KUBECONFIG}."
  echo
  echo "Security note: The server above uses a self signed certificate.  This is"
  echo "    subject to \"Man in the middle\" type attacks."
  echo
}

# Perform preparations required to run e2e tests
function prepare-e2e() {
  echo "Rackspace doesn't need special preparations for e2e tests"
}
