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

SCRIPT_DIR=$(CDPATH="" cd $(dirname $0); pwd)

# Use the config file specified in $KUBE_CONFIG_FILE, or default to
# config-default.sh.
source ${SCRIPT_DIR}/azure/${KUBE_CONFIG_FILE-"config-default.sh"}

function detect-minions () {
    ssh_ports=($(eval echo "2200{1..$NUM_MINIONS}"))
    for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
        MINION_NAMES[$i]=$(ssh -i $AZ_SSH_KEY -p ${ssh_ports[$i]} $AZ_CS.cloudapp.net hostname -f)
    done
}

function detect-master () {
    KUBE_MASTER_IP=${AZ_CS}.cloudapp.net
    echo "Using master: $KUBE_MASTER (external IP: $KUBE_MASTER_IP)"
}

function get-password {
    file=${HOME}/.kubernetes_auth
    if [ -e ${file} ]; then
        user=$(cat $file | python -c 'import json,sys;print json.load(sys.stdin)["User"]')
        passwd=$(cat $file | python -c 'import json,sys;print json.load(sys.stdin)["Password"]')
        return
    fi
    user=admin
    passwd=$(python -c 'import string,random; print "".join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(16))')

    # Store password for reuse.
    cat << EOF > ~/.kubernetes_auth
{
  "User": "$user",
  "Password": "$passwd"
}
EOF
    chmod 0600 ~/.kubernetes_auth
}

# Verify prereqs
function verify-prereqs {
    echo "OK"
    # Already done in sourcing config-default, which sources
    # release/azure/config.sh
}

# Instantiate a kubernetes cluster
function kube-up {
    KUBE_TEMP=$(mktemp -d -t kubernetes.XXXXXX)
    trap "rm -rf ${KUBE_TEMP}" EXIT

    get-password
    echo "Using password: $user:$passwd"
    python $SCRIPT_DIR/../third_party/htpasswd/htpasswd.py -b -c \
        ${KUBE_TEMP}/htpasswd $user $passwd
    HTPASSWD=$(cat ${KUBE_TEMP}/htpasswd)

    # Generate openvpn certs
    echo 01 > ${KUBE_TEMP}/ca.srl
    openssl genrsa -out ${KUBE_TEMP}/ca.key
    openssl req -new -x509 -days 1095 \
        -key ${KUBE_TEMP}/ca.key \
        -out ${KUBE_TEMP}/ca.crt \
        -subj "/CN=openvpn-ca"
    openssl genrsa -out ${KUBE_TEMP}/server.key
    openssl req -new \
        -key ${KUBE_TEMP}/server.key \
        -out ${KUBE_TEMP}/server.csr \
        -subj "/CN=server"
    openssl x509 -req -days 1095 \
        -in ${KUBE_TEMP}/server.csr \
        -CA ${KUBE_TEMP}/ca.crt \
        -CAkey ${KUBE_TEMP}/ca.key \
        -CAserial ${KUBE_TEMP}/ca.srl \
        -out ${KUBE_TEMP}/server.crt
    for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
        openssl genrsa -out ${KUBE_TEMP}/${MINION_NAMES[$i]}.key
        openssl req -new \
            -key ${KUBE_TEMP}/${MINION_NAMES[$i]}.key \
            -out ${KUBE_TEMP}/${MINION_NAMES[$i]}.csr \
            -subj "/CN=${MINION_NAMES[$i]}"
        openssl x509 -req -days 1095 \
            -in ${KUBE_TEMP}/${MINION_NAMES[$i]}.csr \
            -CA ${KUBE_TEMP}/ca.crt \
            -CAkey ${KUBE_TEMP}/ca.key \
            -CAserial ${KUBE_TEMP}/ca.srl \
            -out ${KUBE_TEMP}/${MINION_NAMES[$i]}.crt
    done

    # Build up start up script for master
    (
        echo "#!/bin/bash"
        echo "MASTER_NAME=${MASTER_NAME}"
        echo "MASTER_RELEASE_TAR=${FULL_URL}"
        echo "MASTER_HTPASSWD='${HTPASSWD}'"
        echo "CA_CRT=\"$(cat ${KUBE_TEMP}/ca.crt)\""
        echo "SERVER_CRT=\"$(cat ${KUBE_TEMP}/server.crt)\""
        echo "SERVER_KEY=\"$(cat ${KUBE_TEMP}/server.key)\""
        grep -v "^#" $SCRIPT_DIR/azure/templates/download-release.sh
        grep -v "^#" $SCRIPT_DIR/azure/templates/salt-master.sh
    ) > ${KUBE_TEMP}/master-start.sh

    echo "Starting VMs"

    if [ ! -f $AZ_SSH_KEY ]; then
        ssh-keygen -f $AZ_SSH_KEY -N ''
    fi

    if [ ! -f $AZ_SSH_CERT ]; then
        openssl req -new -x509 -days 1095 -key $AZ_SSH_KEY -out $AZ_SSH_CERT \
            -subj "/CN=azure-ssh-key"
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
            echo "CA_CRT=\"$(cat ${KUBE_TEMP}/ca.crt)\""
            echo "CLIENT_CRT=\"$(cat ${KUBE_TEMP}/${MINION_NAMES[$i]}.crt)\""
            echo "CLIENT_KEY=\"$(cat ${KUBE_TEMP}/${MINION_NAMES[$i]}.key)\""
            grep -v "^#" $SCRIPT_DIR/azure/templates/salt-minion.sh
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
            echo "Docker failed to install on ${MINION_NAMES[$i]}. Your cluster is unlikely to work correctly."
            echo "Please run ./cluster/kube-down.sh and re-create the cluster. (sorry!)"
            exit 1
        fi

        # Make sure the kubelet is running
        ssh -i $AZ_SSH_KEY -p ${ssh_ports[$i]} $AZ_CS.cloudapp.net /etc/init.d/kubelet status
        if [ "$?" != "0" ]; then
            echo "Kubelet failed to install on ${MINION_NAMES[$i]}. Your cluster is unlikely to work correctly."
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
}

# Delete a kubernetes cluster
function kube-down {
    echo "Bringing down cluster"
    set +e
    azure vm delete $MASTER_NAME -b -q
    for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
        azure vm delete ${MINION_NAMES[$i]} -b -q
    done
}

# # Update a kubernetes cluster with latest source
# function kube-push {

#   # Find the release to use.  Generally it will be passed when doing a 'prod'
#   # install and will default to the release/config.sh version when doing a
#   # developer up.
#   find-release $1

#   # Detect the project into $PROJECT
#   detect-master

#   (
#     echo MASTER_RELEASE_TAR=$RELEASE_NORMALIZED/master-release.tgz
#     grep -v "^#" $(dirname $0)/templates/download-release.sh
#     echo "echo Executing configuration"
#     echo "sudo salt '*' mine.update"
#     echo "sudo salt --force-color '*' state.highstate"
#   ) | gcutil ssh --project ${PROJECT} --zone ${ZONE} $KUBE_MASTER bash

#   get-password

#   echo "Kubernetes cluster is updated.  Access the master at:"
#   echo
#   echo "  https://${user}:${passwd}@${KUBE_MASTER_IP}"
#   echo

# }

# # Execute prior to running tests to build a release if required for env
# function test-build-release {
#   # Build source
#   ${KUBE_REPO_ROOT}/hack/build-go.sh
#   # Make a release
#   $(dirname $0)/../release/release.sh
# }

# # Execute prior to running tests to initialize required structure
# function test-setup {

#   # Detect the project into $PROJECT if it isn't set
#   # gce specific
#   detect-project

#   if [[ ${ALREADY_UP} -ne 1 ]]; then
#     # Open up port 80 & 8080 so common containers on minions can be reached
#     gcutil addfirewall \
#       --norespect_terminal_width \
#       --project ${PROJECT} \
#       --target_tags ${MINION_TAG} \
#       --allowed tcp:80,tcp:8080 \
#       --network ${NETWORK} \
#       ${MINION_TAG}-${INSTANCE_PREFIX}-http-alt
#   fi

# }

# # Execute after running tests to perform any required clean-up
# function test-teardown {
#   echo "Shutting down test cluster in background."
#   gcutil deletefirewall  \
#     --project ${PROJECT} \
#     --norespect_terminal_width \
#     --force \
#     ${MINION_TAG}-${INSTANCE_PREFIX}-http-alt || true > /dev/null
#   $(dirname $0)/../cluster/kube-down.sh > /dev/null
# }
