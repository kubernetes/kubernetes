#!/usr/bin/env bash

# Copyright 2014 The Kubernetes Authors All rights reserved.
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
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"

AZURE_ROOT="$DIR"
KUBE_ROOT="$DIR/../.."
KUBE_CONFIG_FILE="${KUBE_CONFIG_FILE:-"${AZURE_ROOT}/config-default.sh"}"
source "${KUBE_CONFIG_FILE}"
source "${KUBE_ROOT}/cluster/common.sh"


function verify-prereqs() {
    required_binaries=("docker")

    for rb in "${required_binaries[@]}"; do
    if ! which "$rb" > /dev/null 2>&1; then
        echo "Couldn't find $rb in PATH"
        exit 1
    fi
    done

    if ! "$KUBE_ROOT/cluster/kubectl.sh" >/dev/null 2>&1 ; then
        echo "kubectl is unavailable. Ensure $KUBE_ROOT/cluster/kubectl.sh runs with a successful exit."
        exit 1
    fi
}

function azure-ensure-config() {
    if [[ -z "${AZURE_SUBSCRIPTION_ID:-}" ]]; then
        echo "AZURE_SUBSCRIPTION_ID must be set"
        exit 1
    fi
    if [[ -z "${AZURE_TENANT_ID:-}" ]]; then
        echo "AZURE_TENANT_ID must be set"
        exit 1
    fi
    export AZURE_DEPLOY_ID="${AZURE_DEPLOY_ID:-kube-$(date +"%Y%m%d-%H%M%S")}"
    export AZURE_LOCATION="${AZURE_LOCATION:-westus}"
    export AZURE_MASTER_SIZE="${AZURE_MASTER_SIZE:-"Standard_A1"}"
    export AZURE_NODE_SIZE="${AZURE_NODE_SIZE:-"Standard_A1"}"
    export NODE_COUNT="${NODE_COUNT:-3}"
    export AZURE_USERNAME="${AZURE_USERNAME:-"kube"}"

    export AZURE_OUTPUT_RELDIR="_deployments/${AZURE_DEPLOY_ID}"
    export AZURE_OUTPUT_DIR="${DIR}/${AZURE_OUTPUT_RELDIR}"
    mkdir -p "${AZURE_OUTPUT_DIR}"

    case "${AZURE_AUTH_METHOD:-}" in
        "client_secret")
            if [[ -z "${AZURE_CLIENT_ID}" ]]; then
                echo "AZURE_CLIENT_ID must be set"
                exit 1
            fi
            if [[ -z "${AZURE_CLIENT_SECRET}" ]]; then
                echo "AZURE_CLIENT_SECRET must be set"
                exit 1
            fi
            ;;
        "device")
            ;;
        "")
            echo "AZURE_AUTH_METHOD not set, assuming \"device\"."
            echo " - This will be interactive."
            echo " - Set AZURE_AUTH_METHOD=client_secret and AZURE_CLIENT_ID/AZURE_CLIENT_SECRET to avoid the prompt"
            export AZURE_AUTH_METHOD="device"
            ;;
        *)
            echo "AZURE_AUTH_METHOD is an unsupported value: ${AZURE_AUTH_METHOD}"
            exit 1
            ;;
    esac
}

function azure-deploy(){
    case "${AZURE_AUTH_METHOD}" in
        "client_secret")
            auth_params1="--client-secret=${AZURE_CLIENT_SECRET}"
            auth_params2="--client-id=${AZURE_CLIENT_ID}"
            ;;
        "device")
            auth_params1=()
            auth_params2=()
            ;;
    esac
    docker run -it \
        --user "$(id -u)" \
        -v "${AZURE_OUTPUT_DIR}:/opt/azkube/${AZURE_OUTPUT_RELDIR}" \
        colemickens/azkube:v0.0.1 /opt/azkube/azkube deploy \
            --kubernetes-hyperkube-spec="${AZURE_HYPERKUBE_SPEC}" \
            --deployment-name="${AZURE_DEPLOY_ID}" \
            --location="${AZURE_LOCATION}" \
            --tenant-id="${AZURE_TENANT_ID}" \
            --subscription-id="${AZURE_SUBSCRIPTION_ID}" \
            --auth-method="${AZURE_AUTH_METHOD}" "${auth_params1}" "${auth_params2}" \
            --master-size="${AZURE_MASTER_SIZE}" \
            --node-size="${AZURE_NODE_SIZE}" \
            --node-count="${NODE_COUNT}" \
            --username="${AZURE_USERNAME}" \
            --output-directory="/opt/azkube/${AZURE_OUTPUT_RELDIR}"
}

function kube-up {
    date_start="$(date)"
    echo "++> AZURE KUBE-UP STARTED: ${date_start}"

    export AZURE_HYPERKUBE_SPEC="gcr.io/google_containers/hyperkube:v1.1.8"

    verify-prereqs
    azure-ensure-config
    azure-deploy

    kubectl config set-cluster "${AZURE_DEPLOY_ID}" --server="https://${AZURE_DEPLOY_ID}.${AZURE_LOCATION}.cloudapp.azure.com:6443/" --certificate-authority="${AZURE_OUTPUT_DIR}/ca.crt" --api-version="v1"
    kubectl config set-credentials "${AZURE_DEPLOY_ID}_user" --client-certificate="${AZURE_OUTPUT_DIR}/client.crt" --client-key="${AZURE_OUTPUT_DIR}/client.key"
    kubectl config set-context "${AZURE_DEPLOY_ID}" --cluster="${AZURE_DEPLOY_ID}" --user="${AZURE_DEPLOY_ID}_user"
    kubectl config use-context "${AZURE_DEPLOY_ID}"

    date_end="$(date)"
    d1="$(date -d "${date_start}" +%s)"
    d2="$(date -d "${date_end}" +%s)"
    duration="$(( (d2 - d1) ))"
    duration_secs="$(( (d2 - d1) )) seconds"
    duration_mins="$(( ((d2 - d1)+60) / 60 )) minutes"
    if [[ ! -z "$(which awk)" ]]; then
        duration_mins=$(awk "BEGIN {printf \"%.2f\",${duration}/60}")
    fi

    echo "++> AZURE KUBE-UP FINISHED: ${date_end}"
    echo "++> AZURE KUBE-UP TIME ELAPSED: ${duration_secs} (${duration_mins})"
}

function kube-down {
    verify-prereqs

    echo "Bringing down cluster"
    echo
    echo "You must do this manually (for now)!"
    echo "This can be done with:"
    echo "   azure_call group delete ${AZ_RESOURCE_GROUP}"
}

