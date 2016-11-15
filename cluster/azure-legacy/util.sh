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

# A library of helper functions and constant for the local config.

# Use the config file specified in $KUBE_CONFIG_FILE, or default to
# config-default.sh.

set -e

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
source "${KUBE_ROOT}/cluster/azure-legacy/${KUBE_CONFIG_FILE-"config-default.sh"}"
source "${KUBE_ROOT}/cluster/common.sh"


function prepare-e2e() {
  # (e2e script runs detect-project, I don't think we need to anything)
  # Note: we can't print anything here, or else the test tools will break with the extra output
  return
}

function azure_call {
    local -a params=()
    local param
    # the '... in "$@"' is implicit on a for, so doesn't need to be stated.
    for param; do
        params+=("${param}")
    done
    local rc=0
    local stderr
    local count=0
    while [[ count -lt 10 ]]; do
        stderr=$(azure "${params[@]}" 2>&1 >&3) && break
        rc=$?
        if [[ "${stderr}" != *"getaddrinfo ENOTFOUND"* ]]; then
            break
        fi
        count=$(($count + 1))
    done 3>&1
    if [[ "${rc}" -ne 0 ]]; then
        echo "${stderr}" >&2
        return "${rc}"
    fi
}

function json_val () {
    python -c 'import json,sys;obj=json.load(sys.stdin);print obj'$1'';
}

# Verify prereqs
function verify-prereqs {
    if [[ -z "$(which azure)" ]]; then
        echo "Couldn't find azure in PATH"
        echo "  please install with 'npm install azure-cli'"
        exit 1
    fi

    if [[ -z "$(azure_call account list | grep true)" ]]; then
        echo "Default azure account not set"
        echo "  please set with 'azure account set'"
        exit 1
    fi

    account=$(azure_call account list | grep true)
    if which md5 > /dev/null 2>&1; then
        AZ_HSH=$(md5 -q -s "$account")
    else
        AZ_HSH=$(echo -n "$account" | md5sum)
    fi

    AZ_HSH=${AZ_HSH:0:7}
    AZ_STG=kube$AZ_HSH
    echo "==> AZ_STG: $AZ_STG"

    AZ_CS="$AZ_CS_PREFIX-$AZ_HSH"
    echo "==> AZ_CS: $AZ_CS"

    CONTAINER=kube-$TAG
    echo "==> CONTAINER: $CONTAINER"
}

# Create a temp dir that'll be deleted at the end of this bash session.
#
# Vars set:
#   KUBE_TEMP
function ensure-temp-dir {
    if [[ -z ${KUBE_TEMP-} ]]; then
        KUBE_TEMP=$(mktemp -d -t kubernetes.XXXXXX)
        trap 'rm -rf "${KUBE_TEMP}"' EXIT
    fi
}

# Take the local tar files and upload them to Azure Storage.  They will then be
# downloaded by the master as part of the start up script for the master.
#
# Assumed vars:
#   SERVER_BINARY_TAR
#   SALT_TAR
# Vars set:
#   SERVER_BINARY_TAR_URL
#   SALT_TAR_URL
function upload-server-tars() {
    SERVER_BINARY_TAR_URL=
    SALT_TAR_URL=

    echo "==> SERVER_BINARY_TAR: $SERVER_BINARY_TAR"
    echo "==> SALT_TAR: $SALT_TAR"

    echo "+++ Staging server tars to Azure Storage: $AZ_STG"
    local server_binary_url="${SERVER_BINARY_TAR##*/}"
    local salt_url="${SALT_TAR##*/}"

    SERVER_BINARY_TAR_URL="https://${AZ_STG}.blob.core.windows.net/$CONTAINER/$server_binary_url"
    SALT_TAR_URL="https://${AZ_STG}.blob.core.windows.net/$CONTAINER/$salt_url"

    echo "==> SERVER_BINARY_TAR_URL: $SERVER_BINARY_TAR_URL"
    echo "==> SALT_TAR_URL: $SALT_TAR_URL"

    echo "--> Checking storage exists..."
    if [[ -z "$(azure_call storage account show $AZ_STG 2>/dev/null | \
    grep data)" ]]; then
        echo "--> Creating storage..."
        azure_call storage account create -l "$AZ_LOCATION" $AZ_STG --type LRS
    fi

    echo "--> Getting storage key..."
    stg_key=$(azure_call storage account keys list $AZ_STG --json | \
        json_val '["primaryKey"]')

    echo "--> Checking storage container exists..."
    if [[ -z "$(azure_call storage container show -a $AZ_STG -k "$stg_key" \
      $CONTAINER 2>/dev/null | grep data)" ]]; then
        echo "--> Creating storage container..."
        azure_call storage container create \
            -a $AZ_STG \
            -k "$stg_key" \
            -p Blob \
            $CONTAINER
    fi

    echo "--> Checking server binary exists in the container..."
    if [[ -n "$(azure_call storage blob show -a $AZ_STG -k "$stg_key" \
      $CONTAINER $server_binary_url 2>/dev/null | grep data)" ]]; then
        echo "--> Deleting server binary in the container..."
        azure_call storage blob delete \
            -a $AZ_STG \
            -k "$stg_key" \
            $CONTAINER \
            $server_binary_url
    fi

    echo "--> Uploading server binary to the container..."
    azure_call storage blob upload \
        -a $AZ_STG \
        -k "$stg_key" \
        $SERVER_BINARY_TAR \
        $CONTAINER \
        $server_binary_url

    echo "--> Checking salt data exists in the container..."
    if [[ -n "$(azure_call storage blob show -a $AZ_STG -k "$stg_key" \
      $CONTAINER $salt_url 2>/dev/null | grep data)" ]]; then
        echo "--> Deleting salt data in the container..."
        azure_call storage blob delete \
            -a $AZ_STG \
            -k "$stg_key" \
            $CONTAINER \
            $salt_url
    fi

    echo "--> Uploading salt data to the container..."
    azure_call storage blob upload \
        -a $AZ_STG \
        -k "$stg_key" \
        $SALT_TAR \
        $CONTAINER \
        $salt_url
}

# Detect the information about the minions
#
# Assumed vars:
#   MINION_NAMES
#   ZONE
# Vars set:
#
function detect-minions () {
    if [[ -z "$AZ_CS" ]]; then
        verify-prereqs-local
    fi
    ssh_ports=($(eval echo "2200{1..$NUM_MINIONS}"))
    for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
        MINION_NAMES[$i]=$(ssh -oStrictHostKeyChecking=no -i $AZ_SSH_KEY -p ${ssh_ports[$i]} $AZ_CS.cloudapp.net hostname -f)
    done
}

# Detect the IP for the master
#
# Assumed vars:
#   MASTER_NAME
#   ZONE
# Vars set:
#   KUBE_MASTER
#   KUBE_MASTER_IP
function detect-master () {
    if [[ -z "$AZ_CS" ]]; then
        verify-prereqs-local
    fi

    KUBE_MASTER=${MASTER_NAME}
    KUBE_MASTER_IP="${AZ_CS}.cloudapp.net"
    echo "Using master: $KUBE_MASTER (external IP: $KUBE_MASTER_IP)"
}

# Instantiate a kubernetes cluster
#
# Assumed vars
#   KUBE_ROOT
#   <Various vars set in config file>
function kube-up {
    # Make sure we have the tar files staged on Azure Storage
    find-release-tars
    upload-server-tars

    ensure-temp-dir

    gen-kube-basicauth
    python "${KUBE_ROOT}/third_party/htpasswd/htpasswd.py" \
        -b -c "${KUBE_TEMP}/htpasswd" "$KUBE_USER" "$KUBE_PASSWORD"
    local htpasswd
    htpasswd=$(cat "${KUBE_TEMP}/htpasswd")

    # Generate openvpn certs
    echo "--> Generating openvpn certs"
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

    KUBE_MASTER_IP="${AZ_CS}.cloudapp.net"

    # Build up start up script for master
    echo "--> Building up start up script for master"
    (
        echo "#!/bin/bash"
        echo "CA_CRT=\"$(cat ${KUBE_TEMP}/ca.crt)\""
        echo "SERVER_CRT=\"$(cat ${KUBE_TEMP}/server.crt)\""
        echo "SERVER_KEY=\"$(cat ${KUBE_TEMP}/server.key)\""
        echo "mkdir -p /var/cache/kubernetes-install"
        echo "cd /var/cache/kubernetes-install"
        echo "readonly MASTER_NAME='${MASTER_NAME}'"
        echo "readonly INSTANCE_PREFIX='${INSTANCE_PREFIX}'"
        echo "readonly NODE_INSTANCE_PREFIX='${INSTANCE_PREFIX}-minion'"
        echo "readonly SERVER_BINARY_TAR_URL='${SERVER_BINARY_TAR_URL}'"
        echo "readonly SALT_TAR_URL='${SALT_TAR_URL}'"
        echo "readonly MASTER_HTPASSWD='${htpasswd}'"
        echo "readonly SERVICE_CLUSTER_IP_RANGE='${SERVICE_CLUSTER_IP_RANGE}'"
        echo "readonly ADMISSION_CONTROL='${ADMISSION_CONTROL:-}'"
        echo "readonly KUBE_USER='${KUBE_USER}'"
        echo "readonly KUBE_PASSWORD='${KUBE_PASSWORD}'"
        echo "readonly KUBE_MASTER_IP='${KUBE_MASTER_IP}'"
        grep -v "^#" "${KUBE_ROOT}/cluster/azure-legacy/templates/common.sh"
        grep -v "^#" "${KUBE_ROOT}/cluster/azure-legacy/templates/create-dynamic-salt-files.sh"
        grep -v "^#" "${KUBE_ROOT}/cluster/azure-legacy/templates/create-kubeconfig.sh"
        grep -v "^#" "${KUBE_ROOT}/cluster/azure-legacy/templates/download-release.sh"
        grep -v "^#" "${KUBE_ROOT}/cluster/azure-legacy/templates/salt-master.sh"
    ) > "${KUBE_TEMP}/master-start.sh"

    if [[ ! -f $AZ_SSH_KEY ]]; then
        ssh-keygen -f $AZ_SSH_KEY -N ''
    fi

    if [[ ! -f $AZ_SSH_CERT ]]; then
        openssl req -new -x509 -days 1095 -key $AZ_SSH_KEY -out $AZ_SSH_CERT \
            -subj "/CN=azure-ssh-key"
    fi

    if [[ -z "$(azure_call network vnet show "$AZ_VNET" 2>/dev/null | grep data)" ]]; then
        echo error create vnet $AZ_VNET with subnet $AZ_SUBNET
        exit 1
    fi

    echo "--> Starting VM"
    azure_call vm create \
        -z "$MASTER_SIZE" \
        -w "$AZ_VNET" \
        -n $MASTER_NAME \
        -l "$AZ_LOCATION" \
        -t $AZ_SSH_CERT \
        -e 22000 -P \
        -d ${KUBE_TEMP}/master-start.sh \
        -b $AZ_SUBNET \
        $AZ_CS $AZ_IMAGE $USER

    ssh_ports=($(eval echo "2200{1..$NUM_MINIONS}"))

    #Build up start up script for minions
    echo "--> Building up start up script for minions"
    for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
        (
            echo "#!/bin/bash"
            echo "MASTER_NAME='${MASTER_NAME}'"
            echo "CA_CRT=\"$(cat ${KUBE_TEMP}/ca.crt)\""
            echo "CLIENT_CRT=\"$(cat ${KUBE_TEMP}/${MINION_NAMES[$i]}.crt)\""
            echo "CLIENT_KEY=\"$(cat ${KUBE_TEMP}/${MINION_NAMES[$i]}.key)\""
            echo "MINION_IP_RANGE='${MINION_IP_RANGES[$i]}'"
            echo "readonly KUBE_USER='${KUBE_USER}'"
            echo "readonly KUBE_PASSWORD='${KUBE_PASSWORD}'"
            echo "readonly KUBE_MASTER_IP='${KUBE_MASTER_IP}'"
            grep -v "^#" "${KUBE_ROOT}/cluster/azure-legacy/templates/common.sh"
            grep -v "^#" "${KUBE_ROOT}/cluster/azure-legacy/templates/create-kubeconfig.sh"
            grep -v "^#" "${KUBE_ROOT}/cluster/azure-legacy/templates/salt-minion.sh"
        ) > "${KUBE_TEMP}/minion-start-${i}.sh"

        echo "--> Starting VM"
        azure_call vm create \
            -z "$MINION_SIZE" \
            -c -w "$AZ_VNET" \
            -n ${MINION_NAMES[$i]} \
            -l "$AZ_LOCATION" \
            -t $AZ_SSH_CERT \
            -e ${ssh_ports[$i]} -P \
            -d ${KUBE_TEMP}/minion-start-${i}.sh \
            -b $AZ_SUBNET \
            $AZ_CS $AZ_IMAGE $USER
    done

    echo "--> Creating endpoint"
    azure_call vm endpoint create $MASTER_NAME 443

    detect-master > /dev/null

    echo "==> KUBE_MASTER_IP: ${KUBE_MASTER_IP}"

    echo "Waiting for cluster initialization."
    echo
    echo "  This will continually check to see if the API for kubernetes is reachable."
    echo "  This might loop forever if there was some uncaught error during start"
    echo "  up."
    echo

    until curl --insecure --user "${KUBE_USER}:${KUBE_PASSWORD}" --max-time 5 \
        --fail --output /dev/null --silent "https://${KUBE_MASTER_IP}/healthz"; do
        printf "."
        sleep 2
    done

    printf "\n"
    echo "Kubernetes cluster created."

    export CONTEXT="azure_${INSTANCE_PREFIX}"
    create-kubeconfig
    export KUBE_CERT="/tmp/$RANDOM-kubecfg.crt"
    export KUBE_KEY="/tmp/$RANDOM-kubecfg.key"
    export CA_CERT="/tmp/$RANDOM-kubernetes.ca.crt"

    # TODO: generate ADMIN (and KUBELET) tokens and put those in the master's
    # config file.  Distribute the same way the htpasswd is done.
(umask 077
    ssh -oStrictHostKeyChecking=no -i $AZ_SSH_KEY -p 22000 $AZ_CS.cloudapp.net \
        sudo cat /srv/kubernetes/kubecfg.crt >"${KUBE_CERT}" 2>/dev/null
    ssh -oStrictHostKeyChecking=no -i $AZ_SSH_KEY -p 22000 $AZ_CS.cloudapp.net \
        sudo cat /srv/kubernetes/kubecfg.key >"${KUBE_KEY}" 2>/dev/null
    ssh -oStrictHostKeyChecking=no -i $AZ_SSH_KEY -p 22000 $AZ_CS.cloudapp.net \
        sudo cat /srv/kubernetes/ca.crt >"${CA_CERT}" 2>/dev/null
)

    echo "Sanity checking cluster..."
    echo
    echo "  This will continually check the minions to ensure docker is"
    echo "  installed. This is usually a good indicator that salt has"
    echo "  successfully  provisioned. This might loop forever if there was"
    echo "  some uncaught error during start up."
    echo
    # Basic sanity checking
    for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
        # Make sure docker is installed
        echo "--> Making sure docker is installed on ${MINION_NAMES[$i]}."
        until ssh -oStrictHostKeyChecking=no -i $AZ_SSH_KEY -p ${ssh_ports[$i]} \
            $AZ_CS.cloudapp.net which docker > /dev/null 2>&1; do
            printf "."
            sleep 2
        done
    done

    sleep 60
    KUBECONFIG_NAME="kubeconfig"
    KUBECONFIG="${HOME}/.kube/config"
    echo "Distributing kubeconfig for kubelet to master kubelet"
    scp -oStrictHostKeyChecking=no -i $AZ_SSH_KEY -P 22000 ${KUBECONFIG} \
        $AZ_CS.cloudapp.net:${KUBECONFIG_NAME}
    ssh -oStrictHostKeyChecking=no -i $AZ_SSH_KEY -p 22000 $AZ_CS.cloudapp.net \
        sudo cp ${KUBECONFIG_NAME} /var/lib/kubelet/${KUBECONFIG_NAME}
    ssh -oStrictHostKeyChecking=no -i $AZ_SSH_KEY -p 22000 $AZ_CS.cloudapp.net \
        sudo service kubelet restart

    echo "Distributing kubeconfig for kubelet to all minions"
    for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
        scp -oStrictHostKeyChecking=no -i $AZ_SSH_KEY -P ${ssh_ports[$i]} ${KUBECONFIG} \
            $AZ_CS.cloudapp.net:${KUBECONFIG_NAME}
        ssh -oStrictHostKeyChecking=no -i $AZ_SSH_KEY -p ${ssh_ports[$i]} $AZ_CS.cloudapp.net \
            sudo cp ${KUBECONFIG_NAME} /var/lib/kubelet/${KUBECONFIG_NAME}
        ssh -oStrictHostKeyChecking=no -i $AZ_SSH_KEY -p ${ssh_ports[$i]} $AZ_CS.cloudapp.net \
            sudo cp ${KUBECONFIG_NAME} /var/lib/kube-proxy/${KUBECONFIG_NAME}
        ssh -oStrictHostKeyChecking=no -i $AZ_SSH_KEY -p ${ssh_ports[$i]} $AZ_CS.cloudapp.net \
            sudo service kubelet restart
        ssh -oStrictHostKeyChecking=no -i $AZ_SSH_KEY -p ${ssh_ports[$i]} $AZ_CS.cloudapp.net \
            sudo killall kube-proxy
    done

    # ensures KUBECONFIG is set
    get-kubeconfig-basicauth
    echo
    echo "Kubernetes cluster is running.  The master is running at:"
    echo
    echo "  https://${KUBE_MASTER_IP}"
    echo
    echo "The user name and password to use is located in ${KUBECONFIG}."
    echo
}

# Delete a kubernetes cluster
function kube-down {
    echo "Bringing down cluster"

    set +e
    azure_call vm delete $MASTER_NAME -b -q
    for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
        azure_call vm delete ${MINION_NAMES[$i]} -b -q
    done

    wait
}

# Update a kubernetes cluster with latest source
#function kube-push {
#  detect-project
#  detect-master

# Make sure we have the tar files staged on Azure Storage
#  find-release-tars
#  upload-server-tars

#  (
#    echo "#! /bin/bash"
#    echo "mkdir -p /var/cache/kubernetes-install"
#    echo "cd /var/cache/kubernetes-install"
#    echo "readonly SERVER_BINARY_TAR_URL='${SERVER_BINARY_TAR_URL}'"
#    echo "readonly SALT_TAR_URL='${SALT_TAR_URL}'"
#    grep -v "^#" "${KUBE_ROOT}/cluster/azure/templates/common.sh"
#    grep -v "^#" "${KUBE_ROOT}/cluster/azure/templates/download-release.sh"
#    echo "echo Executing configuration"
#    echo "sudo salt '*' mine.update"
#    echo "sudo salt --force-color '*' state.highstate"
#   ) | gcutil ssh --project "$PROJECT" --zone "$ZONE" "$KUBE_MASTER" sudo bash

#  get-kubeconfig-basicauth

#  echo
#  echo "Kubernetes cluster is running.  The master is running at:"
#  echo
#  echo "  https://${KUBE_MASTER_IP}"
# echo
#  echo "The user name and password to use is located in ${KUBECONFIG:-$DEFAULT_KUBECONFIG}."
#  echo

#}

# -----------------------------------------------------------------------------
# Cluster specific test helpers

# Execute prior to running tests to build a release if required for env.
#
# Assumed Vars:
#   KUBE_ROOT
function test-build-release {
    # Make a release
    "${KUBE_ROOT}/build-tools/release.sh"
}

# SSH to a node by name ($1) and run a command ($2).
function ssh-to-node {
    local node="$1"
    local cmd="$2"
    ssh --ssh_arg "-o LogLevel=quiet" "${node}" "${cmd}"
}

# Restart the kube-proxy on a node ($1)
function restart-kube-proxy {
    ssh-to-node "$1" "sudo /etc/init.d/kube-proxy restart"
}

# Restart the kube-proxy on the master ($1)
function restart-apiserver {
    ssh-to-node "$1" "sudo /etc/init.d/kube-apiserver restart"
}

function test-setup {
    "${KUBE_ROOT}/cluster/kube-up.sh"
}

function test-teardown {
    "${KUBE_ROOT}/cluster/kube-down.sh"
}
