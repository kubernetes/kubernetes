#!/bin/bash

# Copyright 2014 Google Inc. All rights reserved.
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

# Use the config file specified in $LMKTFY_CONFIG_FILE, or default to
# config-default.sh.
LMKTFY_ROOT=$(dirname "${BASH_SOURCE}")/../..
source $(dirname ${BASH_SOURCE})/${LMKTFY_CONFIG_FILE-"config-default.sh"}

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

# Ensure that we have a password created for validating to the master.  Will
# read from $HOME/.lmktfyrnetres_auth if available.
#
# Vars set:
#   LMKTFY_USER
#   LMKTFY_PASSWORD
get-password() {
  local file="$HOME/.lmktfy_auth"
  if [[ -r "$file" ]]; then
    LMKTFY_USER=$(cat "$file" | python -c 'import json,sys;print json.load(sys.stdin)["User"]')
    LMKTFY_PASSWORD=$(cat "$file" | python -c 'import json,sys;print json.load(sys.stdin)["Password"]')
    return
  fi
  LMKTFY_USER=admin
  LMKTFY_PASSWORD=$(python -c 'import string,random; print "".join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(16))')

  # Store password for reuse.
  cat << EOF > "$file"
{
  "User": "$LMKTFY_USER",
  "Password": "$LMKTFY_PASSWORD"
}
EOF
  chmod 0600 "$file"
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

find-release-tars() {
  SERVER_BINARY_TAR="${LMKTFY_ROOT}/server/lmktfy-server-linux-amd64.tar.gz"
  RELEASE_DIR="${LMKTFY_ROOT}/server/"
  if [[ ! -f "$SERVER_BINARY_TAR" ]]; then
    SERVER_BINARY_TAR="${LMKTFY_ROOT}/_output/release-tars/lmktfy-server-linux-amd64.tar.gz"
    RELEASE_DIR="${LMKTFY_ROOT}/_output/release-tars/"
  fi
  if [[ ! -f "$SERVER_BINARY_TAR" ]]; then
    echo "!!! Cannot find lmktfy-server-linux-amd64.tar.gz"
    exit 1
  fi
}

rackspace-set-vars() {

  CLOUDFILES_CONTAINER="lmktfy-releases-${OS_USERNAME}"
  CONTAINER_PREFIX=${CONTAINER_PREFIX-devel/}
  find-release-tars
}

# Retrieves a tempurl from cloudfiles to make the release object publicly accessible temporarily.
find-object-url() {

  rackspace-set-vars

  LMKTFY_TAR=${CLOUDFILES_CONTAINER}/${CONTAINER_PREFIX}/lmktfy-server-linux-amd64.tar.gz

  RELEASE_TMP_URL=$(swiftly -A ${OS_AUTH_URL} -U ${OS_USERNAME} -K ${OS_PASSWORD} tempurl GET ${LMKTFY_TAR})
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

# Copy lmktfy-server-linux-amd64.tar.gz to cloud files object store
copy_dev_tarballs() {

  echo "cluster/rackspace/util.sh: Uploading to Cloud Files"
  ${SWIFTLY_CMD} put -i ${RELEASE_DIR}/lmktfy-server-linux-amd64.tar.gz \
  ${CLOUDFILES_CONTAINER}/${CONTAINER_PREFIX}/lmktfy-server-linux-amd64.tar.gz > /dev/null 2>&1
  
  echo "Release pushed."
}

rax-boot-master() {

  DISCOVERY_URL=$(curl https://discovery.etcd.io/new)
  DISCOVERY_ID=$(echo "${DISCOVERY_URL}" | cut -f 4 -d /)
  echo "cluster/rackspace/util.sh: etcd discovery URL: ${DISCOVERY_URL}"

# Copy cloud-config to LMKTFY_TEMP and work some sed magic
  sed -e "s|DISCOVERY_ID|${DISCOVERY_ID}|" \
      -e "s|CLOUD_FILES_URL|${RELEASE_TMP_URL//&/\\&}|" \
      -e "s|LMKTFY_USER|${LMKTFY_USER}|" \
      -e "s|LMKTFY_PASSWORD|${LMKTFY_PASSWORD}|" \
      -e "s|PORTAL_NET|${PORTAL_NET}|" \
      -e "s|OS_AUTH_URL|${OS_AUTH_URL}|" \
      -e "s|OS_USERNAME|${OS_USERNAME}|" \
      -e "s|OS_PASSWORD|${OS_PASSWORD}|" \
      -e "s|OS_TENANT_NAME|${OS_TENANT_NAME}|" \
      -e "s|OS_REGION_NAME|${OS_REGION_NAME}|" \
      $(dirname $0)/rackspace/cloud-config/master-cloud-config.yaml > $LMKTFY_TEMP/master-cloud-config.yaml


  MASTER_BOOT_CMD="nova boot \
--key-name ${SSH_KEY_NAME} \
--flavor ${LMKTFY_MASTER_FLAVOR} \
--image ${LMKTFY_IMAGE} \
--meta ${MASTER_TAG} \
--meta ETCD=${DISCOVERY_ID} \
--user-data ${LMKTFY_TEMP}/master-cloud-config.yaml \
--config-drive true \
--nic net-id=${NETWORK_UUID} \
${MASTER_NAME}"

  echo "cluster/rackspace/util.sh: Booting ${MASTER_NAME} with following command:"
  echo -e "\t$MASTER_BOOT_CMD"
  $MASTER_BOOT_CMD
}

rax-boot-minions() {

  cp $(dirname $0)/rackspace/cloud-config/minion-cloud-config.yaml \
  ${LMKTFY_TEMP}/minion-cloud-config.yaml

  for (( i=0; i<${#MINION_NAMES[@]}; i++)); do

    sed -e "s|DISCOVERY_ID|${DISCOVERY_ID}|" \
        -e "s|INDEX|$((i + 1))|g" \
        -e "s|CLOUD_FILES_URL|${RELEASE_TMP_URL//&/\\&}|" \
        -e "s|ENABLE_NODE_MONITORING|${ENABLE_NODE_MONITORING:-false}|" \
        -e "s|ENABLE_NODE_LOGGING|${ENABLE_NODE_LOGGING:-false}|" \
        -e "s|LOGGING_DESTINATION|${LOGGING_DESTINATION:-}|" \
        -e "s|ENABLE_CLUSTER_DNS|${ENABLE_CLUSTER_DNS:-false}|" \
        -e "s|DNS_SERVER_IP|${DNS_SERVER_IP:-}|" \
        -e "s|DNS_DOMAIN|${DNS_DOMAIN:-}|" \
    $(dirname $0)/rackspace/cloud-config/minion-cloud-config.yaml > $LMKTFY_TEMP/minion-cloud-config-$(($i + 1)).yaml


    MINION_BOOT_CMD="nova boot \
--key-name ${SSH_KEY_NAME} \
--flavor ${LMKTFY_MINION_FLAVOR} \
--image ${LMKTFY_IMAGE} \
--meta ${MINION_TAG} \
--user-data ${LMKTFY_TEMP}/minion-cloud-config-$(( i +1 )).yaml \
--config-drive true \
--nic net-id=${NETWORK_UUID} \
${MINION_NAMES[$i]}"

    echo "cluster/rackspace/util.sh: Booting ${MINION_NAMES[$i]} with following command:"
    echo -e "\t$MINION_BOOT_CMD"
    $MINION_BOOT_CMD
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

detect-minions() {
  LMKTFY_MINION_IP_ADDRESSES=()
  for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
    local minion_ip=$(nova show --minimal ${MINION_NAMES[$i]} \
      | grep accessIPv4 | awk '{print $4}')
    echo "cluster/rackspace/util.sh: Found ${MINION_NAMES[$i]} at ${minion_ip}"
    LMKTFY_MINION_IP_ADDRESSES+=("${minion_ip}")
  done
  if [ -z "$LMKTFY_MINION_IP_ADDRESSES" ]; then
    echo "cluster/rackspace/util.sh: Could not detect LMKTFY minion nodes.  Make sure you've launched a cluster with 'lmktfy-up.sh'"
    exit 1
  fi

}

detect-master() {
  LMKTFY_MASTER=${MASTER_NAME}

  echo "Waiting for ${MASTER_NAME} IP Address."
  echo
  echo "  This will continually check to see if the master node has an IP address."
  echo

  LMKTFY_MASTER_IP=$(nova show $LMKTFY_MASTER --minimal | grep accessIPv4 | awk '{print $4}')

  while [ "${LMKTFY_MASTER_IP-|}" == "|" ]; do
    LMKTFY_MASTER_IP=$(nova show $LMKTFY_MASTER --minimal | grep accessIPv4 | awk '{print $4}')
    printf "."
    sleep 2
  done

  echo "${LMKTFY_MASTER} IP Address is ${LMKTFY_MASTER_IP}"
}

# $1 should be the network you would like to get an IP address for
detect-master-nova-net() {
  LMKTFY_MASTER=${MASTER_NAME}

  MASTER_IP=$(nova show $LMKTFY_MASTER --minimal | grep $1 | awk '{print $5}')
}

lmktfy-up() {

  SCRIPT_DIR=$(CDPATH="" cd $(dirname $0); pwd)

  rackspace-set-vars
  ensure_dev_container
  copy_dev_tarballs

  # Find the release to use.  Generally it will be passed when doing a 'prod'
  # install and will default to the release/config.sh version when doing a
  # developer up.
  find-object-url

  # Create a temp directory to hold scripts that will be uploaded to master/minions
  LMKTFY_TEMP=$(mktemp -d -t lmktfy.XXXXXX)
  trap "rm -rf ${LMKTFY_TEMP}" EXIT

  get-password
  python $(dirname $0)/../third_party/htpasswd/htpasswd.py -b -c ${LMKTFY_TEMP}/htpasswd $LMKTFY_USER $LMKTFY_PASSWORD
  HTPASSWD=$(cat ${LMKTFY_TEMP}/htpasswd)

  rax-nova-network
  NETWORK_UUID=$(nova network-list | grep -i ${NOVA_NETWORK_LABEL} | awk '{print $2}')

  # create and upload ssh key if necessary
  rax-ssh-key

  echo "cluster/rackspace/util.sh: Starting Cloud Servers"
  rax-boot-master

  rax-boot-minions

  FAIL=0
  for job in `jobs -p`
  do
      wait $job || let "FAIL+=1"
  done
  if (( $FAIL != 0 )); then
    echo "${FAIL} commands failed.  Exiting."
    exit 2
  fi

  detect-master

  echo "Waiting for cluster initialization."
  echo
  echo "  This will continually check to see if the API for lmktfy is reachable."
  echo "  This might loop forever if there was some uncaught error during start"
  echo "  up."
  echo

  #This will fail until apiserver salt is updated
  until $(curl --insecure --user ${LMKTFY_USER}:${LMKTFY_PASSWORD} --max-time 5 \
          --fail --output /dev/null --silent https://${LMKTFY_MASTER_IP}/api/v1beta1/pods); do
      printf "."
      sleep 2
  done

  echo "LMKTFY cluster created."

  # Don't bail on errors, we want to be able to print some info.
  set +e

  detect-minions

  echo "All minions may not be online yet, this is okay."
  echo
  echo "LMKTFY cluster is running.  The master is running at:"
  echo
  echo "  https://${LMKTFY_MASTER_IP}"
  echo
  echo "The user name and password to use is located in ~/.lmktfy_auth."
  echo
  echo "Security note: The server above uses a self signed certificate.  This is"
  echo "    subject to \"Man in the middle\" type attacks."
  echo
}

# Perform preparations required to run e2e tests
function prepare-e2e() {
  echo "Rackspace doesn't need special preparations for e2e tests"
}
