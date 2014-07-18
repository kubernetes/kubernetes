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

# Bring up a Kubernetes cluster.

# exit on any error
set -eu
set -o pipefail
SCRIPT_DIR=$(CDPATH="" cd $(dirname $0); pwd)

source $SCRIPT_DIR/../../release/azure/config.sh
source $SCRIPT_DIR/../util.sh

KUBE_TEMP=$(mktemp -d -t kubernetes.XXXXXX)
trap "rm -rf ${KUBE_TEMP}" EXIT

get-password
echo "Using password: $user:$passwd"
python $SCRIPT_DIR/../../third_party/htpasswd/htpasswd.py -b -c \
    ${KUBE_TEMP}/htpasswd $user $passwd
HTPASSWD=$(cat ${KUBE_TEMP}/htpasswd)

# Build up start up script for master
(
  echo "#!/bin/bash"
  echo "MASTER_NAME=${MASTER_NAME}"
  echo "MASTER_RELEASE_TAR=${FULL_URL}"
  echo "MASTER_HTPASSWD='${HTPASSWD}'"
  grep -v "^#" $SCRIPT_DIR/templates/download-release.sh
  grep -v "^#" $SCRIPT_DIR/templates/salt-master.sh
) > ${KUBE_TEMP}/master-start.sh

echo "Starting VMs"

if [ ! -f $AZ_SSH_KEY ]; then
    ssh-keygen -f $AZ_SSH_KEY -N ''
fi

if [ ! -f $AZ_SSH_CERT ]; then
    openssl req -new -key $AZ_SSH_KEY -out ${KUBE_TEMP}/temp.csr \
        -subj "/C=US/ST=WA/L=Redmond/O=Azure-CLI/CN=Azure"
    openssl req -x509 -key $AZ_SSH_KEY -in ${KUBE_TEMP}/temp.csr \
        -out $AZ_SSH_CERT -days 1095
    rm ${KUBE_TEMP}/temp.csr
fi

if [ -z "$(azure network vnet show $AZ_VNET 2>/dev/null | grep data)" ]; then
    #azure network vnet create with $AZ_SUBNET
    #FIXME not working
    echo error create vnet $AZ_VNET with subnet $AZ_SUBNET
    exit 1
fi

azure vm create \
    -w $AZ_VNET \
    -n $MASTER_NAME \
    -l "$AZ_LOCATION" \
    -t $AZ_SSH_CERT \
    -e 22000 -P \
    -d ${KUBE_TEMP}/master-start.sh \
    -b $AZ_SUBNET \
    $AZ_CS $AZ_IMAGE $USER

ssh_ports=($(eval echo "2200{1..$NUM_MINIONS}"))

for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
    (
        echo "#!/bin/bash"
        echo "MASTER_NAME=${MASTER_NAME}"
        echo "MINION_IP_RANGE=${MINION_IP_RANGES[$i]}"
        grep -v "^#" $SCRIPT_DIR/templates/salt-minion.sh
    ) > ${KUBE_TEMP}/minion-start-${i}.sh

    azure vm create \
        -c -w $AZ_VNET \
        -n ${MINION_NAMES[$i]} \
        -l "$AZ_LOCATION" \
        -t $AZ_SSH_CERT \
        -e ${ssh_ports[$i]} -P \
        -d ${KUBE_TEMP}/minion-start-${i}.sh \
        -b $AZ_SUBNET \
        $AZ_CS $AZ_IMAGE $USER
done

azure vm endpoint create $MASTER_NAME 443

echo "Waiting for cluster initialization."
echo
echo "  This will continually check to see if the API for kubernetes is reachable."
echo "  This might loop forever if there was some uncaught error during start"
echo "  up."
echo

until $(curl --insecure --user ${user}:${passwd} --max-time 5 \
        --fail --output /dev/null --silent https://$AZ_CS.cloudapp.net/api/v1beta1/pods); do
    printf "."
    sleep 2
done

# Basic sanity checking
for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
    # Make sure docker is installed
    ssh -i $AZ_SSH_KEY -p ${ssh_ports[$i]} $AZ_CS.cloudapp.net which docker > /dev/null
    if [ "$?" != "0" ]; then
        echo "Docker failed to install on ${MINION_NAMES[$i]} your cluster is unlikely to work correctly"
        echo "Please run ./cluster/kube-down.sh and re-create the cluster. (sorry!)"
        exit 1
    fi

    # Make sure the kubelet is running
    ssh -i $AZ_SSH_KEY -p ${ssh_ports[$i]} $AZ_CS.cloudapp.net /etc/init.d/kubelet status
    if [ "$?" != "0" ]; then
        echo "Kubelet failed to install on ${MINION_NAMES[$i]} your cluster is unlikely to work correctly"
        echo "Please run ./cluster/kube-down.sh and re-create the cluster. (sorry!)"
        exit 1
    fi
done

echo
echo "Kubernetes cluster is running.  Access the master at:"
echo
echo "  https://${user}:${passwd}@$AZ_CS.cloudapp.net"
echo
echo "Security note: The server above uses a self signed certificate.  This is"
echo "    subject to \"Man in the middle\" type attacks."
