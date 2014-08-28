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

# Use the config file specified in $KUBE_CONFIG_FILE, or default to
# config-default.sh.
source $(dirname ${BASH_SOURCE})/${KUBE_CONFIG_FILE-"config-default.sh"}

verify-prereqs() {
  # Make sure that prerequisites are installed.
  for x in nova; do
    if [ "$(which $x)" == "" ]; then
      echo "cluster/rackspace/util.sh:  Can't find $x in PATH, please fix and retry."
      exit 1
    fi
  done
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

find-object-url() {
  if [ -n "$1" ]; then
    CONTAINER=$1
  else
    local RELEASE_CONFIG_SCRIPT=$(dirname $0)/../release/rackspace/config.sh
    if [ -f $(dirname $0)/../release/rackspace/config.sh ]; then
      . $RELEASE_CONFIG_SCRIPT
    fi
  fi

  TEMP_URL=$(swiftly -A ${OS_AUTH_URL} -U ${OS_USERNAME} -K ${OS_PASSWORD} tempurl GET $1/$2)
  echo "cluster/rackspace/util.sh: Object temp URL:"
  echo -e "\t${TEMP_URL}"

}

rax-boot-master() {

  (
  echo "#! /bin/bash"
  echo "OBJECT_URL=\"${TEMP_URL}\""
  echo "MASTER_HTPASSWD=${HTPASSWD}"
  grep -v "^#" $(dirname $0)/templates/download-release.sh
  ) > ${KUBE_TEMP}/masterStart.sh

# Copy cloud-config to KUBE_TEMP and work some sed magic
  sed -e "s/KUBE_MASTER/$MASTER_NAME/" \
      -e "s/MASTER_HTPASSWD/$HTPASSWD/" \
      $(dirname $0)/cloud-config/master-cloud-config.yaml > $KUBE_TEMP/master-cloud-config.yaml


  MASTER_BOOT_CMD="nova boot \
--key-name ${SSH_KEY_NAME} \
--flavor ${KUBE_MASTER_FLAVOR} \
--image ${KUBE_IMAGE} \
--meta ${MASTER_TAG} \
--user-data ${KUBE_TEMP}/master-cloud-config.yaml \
--config-drive true \
--file /root/masterStart.sh=${KUBE_TEMP}/masterStart.sh \
--nic net-id=${NETWORK_UUID} \
${MASTER_NAME}"
  
  echo "cluster/rackspace/util.sh: Booting ${MASTER_NAME} with following command:"
  echo -e "\t$MASTER_BOOT_CMD"
  $MASTER_BOOT_CMD
}

rax-boot-minions() {

  cp $(dirname $0)/cloud-config/minion-cloud-config.yaml \
  ${KUBE_TEMP}/minion-cloud-config.yaml
  
  for (( i=0; i<${#MINION_NAMES[@]}; i++)); do

    (
      echo "#! /bin/bash"
      echo "MASTER_NAME=${MASTER_IP}"
      echo "MINION_IP_RANGE=${KUBE_NETWORK[$i]}"
      echo "NUM_MINIONS=${RAX_NUM_MINIONS}"
      grep -v "^#" $(dirname $0)/templates/salt-minion.sh
    ) > ${KUBE_TEMP}/minionStart${i}.sh
  
    MINION_BOOT_CMD="nova boot \
--key-name ${SSH_KEY_NAME} \
--flavor ${KUBE_MINION_FLAVOR} \
--image ${KUBE_IMAGE} \
--meta ${MINION_TAG} \
--user-data ${KUBE_TEMP}/minion-cloud-config.yaml \
--config-drive true \
--nic net-id=${NETWORK_UUID} \
--file=/root/minionStart.sh=${KUBE_TEMP}/minionStart${i}.sh \
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
  KUBE_MINION_IP_ADDRESSES=()
  for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
    local minion_ip=$(nova show --minimal ${MINION_NAMES[$i]} \
      | grep accessIPv4 | awk '{print $4}')
    echo "cluster/rackspace/util.sh: Found ${MINION_NAMES[$i]} at ${minion_ip}"
    KUBE_MINION_IP_ADDRESSES+=("${minion_ip}")
  done
  if [ -z "$KUBE_MINION_IP_ADDRESSES" ]; then
    echo "cluster/rackspace/util.sh: Could not detect Kubernetes minion nodes.  Make sure you've launched a cluster with 'kube-up.sh'"
    exit 1
  fi

}

detect-master() {
  KUBE_MASTER=${MASTER_NAME}

  KUBE_MASTER_IP=$(nova show $KUBE_MASTER --minimal | grep accessIPv4 | awk '{print $4}')
}

# $1 should be the network you would like to get an IP address for
detect-master-nova-net() {
  KUBE_MASTER=${MASTER_NAME}

  MASTER_IP=$(nova show $KUBE_MASTER --minimal | grep $1 | awk '{print $5}')
}

kube-up() {
  
  SCRIPT_DIR=$(CDPATH="" cd $(dirname $0); pwd)
  source $(dirname $0)/../gce/util.sh
  source $(dirname $0)/util.sh
  source $(dirname $0)/../../release/rackspace/config.sh
  
  # Find the release to use.  Generally it will be passed when doing a 'prod'
  # install and will default to the release/config.sh version when doing a
  # developer up.
  find-object-url $CONTAINER output/release/$TAR_FILE
  
  # Create a temp directory to hold scripts that will be uploaded to master/minions
  KUBE_TEMP=$(mktemp -d -t kubernetes.XXXXXX)
  trap "rm -rf ${KUBE_TEMP}" EXIT
  
  get-password
  echo "cluster/rackspace/util.sh: Using password: $user:$passwd"
  python $(dirname $0)/../../third_party/htpasswd/htpasswd.py -b -c ${KUBE_TEMP}/htpasswd $user $passwd
  HTPASSWD=$(cat ${KUBE_TEMP}/htpasswd)
  
  rax-nova-network
  NETWORK_UUID=$(nova network-list | grep -i ${NOVA_NETWORK_LABEL} | awk '{print $2}')
  
  # create and upload ssh key if necessary
  rax-ssh-key
  
  echo "cluster/rackspace/util.sh: Starting Cloud Servers"
  rax-boot-master
  
  # a bit of a hack to wait until master is has an IP from the extra network
  echo "cluster/rackspace/util.sh: sleeping 30 seconds"
  sleep 30
  
  detect-master-nova-net $NOVA_NETWORK_LABEL
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

  detect-master > /dev/null

  echo "Waiting for cluster initialization."
  echo
  echo "  This will continually check to see if the API for kubernetes is reachable."
  echo "  This might loop forever if there was some uncaught error during start"
  echo "  up."
  echo
  
  #This will fail until apiserver salt is updated
  #until $(curl --insecure --user ${user}:${passwd} --max-time 5 \
  #        --fail --output /dev/null --silent https://${KUBE_MASTER_IP}/api/v1beta1/pods); do
  #    printf "."
  #    sleep 2
  #done
  
  echo "Kubernetes cluster created."
  echo "Sanity checking cluster..."
  
  sleep 5
  
  # Don't bail on errors, we want to be able to print some info.
  set +e
  sleep 45

  #detect-minions > /dev/null
  detect-minions


  #This will fail until apiserver salt is updated
  # Basic sanity checking
  #for (( i=0; i<${#KUBE_MINION_IP_ADDRESSES[@]}; i++)); do
  #
  #    # Make sure the kubelet is running
  #  if [ "$(curl --insecure --user ${user}:${passwd} https://${KUBE_MASTER_IP}/proxy/minion/${KUBE_MINION_IP_ADDRESSES[$i]}/healthz)" != "ok" ]; then
  #      echo "Kubelet failed to install on ${KUBE_MINION_IP_ADDRESSES[$i]} your cluster is unlikely to work correctly"
  #      echo "Please run ./cluster/kube-down.sh and re-create the cluster. (sorry!)"
  #      exit 1
  #  else
  #    echo "Kubelet is successfully installed on ${MINION_NAMES[$i]}"
  #
  #  fi
  #
  #done
  echo "All minions may not be online yet, this is okay."
  echo
  echo "Kubernetes cluster is running.  Access the master at:"
  echo
  echo "  https://${user}:${passwd}@${KUBE_MASTER_IP}"
  echo
  echo "Security note: The server above uses a self signed certificate.  This is"
  echo "    subject to \"Man in the middle\" type attacks."
}
