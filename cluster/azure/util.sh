#!/usr/bin/env bash

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

KUBE_ROOT="${DIR}/../.."
KUBE_CONFIG_FILE="${KUBE_CONFIG_FILE:-"${DIR}/config-default.sh"}"
source "${KUBE_CONFIG_FILE}"
source "${KUBE_ROOT}/cluster/common.sh"

AZKUBE_VERSION="v0.0.5"
REGISTER_MASTER_KUBELET="true"

function verify-prereqs() {
    required_binaries=("docker" "jq")

    for rb in "${required_binaries[@]}"; do
    if ! which "$rb" > /dev/null 2>&1; then
        echo "Couldn't find ${rb} in PATH"
        exit 1
    fi
    done

    if ! "${KUBE_ROOT}/cluster/kubectl.sh" >/dev/null 2>&1 ; then
        echo "kubectl is unavailable. Ensure ${KUBE_ROOT}/cluster/kubectl.sh runs with a successful exit."
        exit 1
    fi
}

function azure-ensure-config() {
    if [[ -z "${AZURE_SUBSCRIPTION_ID:-}" ]]; then
        echo "AZURE_SUBSCRIPTION_ID must be set"
        exit 1
    fi

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
        "")
            echo "AZURE_AUTH_METHOD not set, assuming \"device\"."
            ;;
        "device" | "")
            echo "This will be interactive. (export AZURE_AUTH_METHOD=client_secret to avoid the prompt)"
            export AZURE_AUTH_METHOD="device"
            ;;
        *)
            echo "AZURE_AUTH_METHOD is an unsupported value: \"${AZURE_AUTH_METHOD}\""
            exit 1
            ;;
    esac
}

function repo-contains-image() {
    registry="$1"
    repo="$2"
    image="$3"
    version="$4"

    prefix="${registry}"
    if [[ "${prefix}" == "docker.io" ]]; then
        prefix="registry.hub.docker.com/v2/repositories"
        tags_json=$(curl "https://registry.hub.docker.com/v2/repositories/${repo}/${image}/tags/${version}/" 2>/dev/null)
        tags_found="$(echo "${tags_json}" | jq ".v2?")"
    elif [[ "${prefix}" == "gcr.io" ]]; then
        tags_json=$(curl "https://gcr.io/v2/${repo}/${image}/tags/list" 2>/dev/null)
        tags_found="$(echo "${tags_json}" | jq ".tags | indices([\"${version}\"]) | any")"
    fi


    if [[ "${tags_found}" == "true" ]]; then
        return 0
    fi

    return 1
}

function ensure-hyperkube() {
    hyperkube="hyperkube-amd64"
    official_image_tag="gcr.io/google_containers/${hyperkube}:${KUBE_GIT_VERSION}"

    if repo-contains-image "gcr.io" "google_containers" "${hyperkube}" "${KUBE_GIT_VERSION}" ; then
        echo "${hyperkube}:${KUBE_GIT_VERSION} was found in the gcr.io/google_containers repository"
        export AZURE_HYPERKUBE_SPEC="${official_image_tag}"
        return 0
    fi

    echo "${hyperkube}:${KUBE_GIT_VERSION} was not found in the gcr.io/google_containers repository"
    if [[ -z "${AZURE_DOCKER_REGISTRY:-}" || -z "${AZURE_DOCKER_REPO:-}" ]]; then
        echo "AZURE_DOCKER_REGISTRY and AZURE_DOCKER_REPO must be set in order to push ${hyperkube}:${KUBE_GIT_VERSION}"
        return 1
    fi

    # check if it is already in the user owned docker hub
    local user_image_tag="${AZURE_DOCKER_REGISTRY}/${AZURE_DOCKER_REPO}/${hyperkube}:${KUBE_GIT_VERSION}"
    if repo-contains-image "${AZURE_DOCKER_REGISTRY}" "${AZURE_DOCKER_REPO}" "${hyperkube}" "${KUBE_GIT_VERSION}" ; then
        echo "${image}:${version} was found in ${repo} (success)"
        export AZURE_HYPERKUBE_SPEC="${user_image_tag}"
        return 0
    fi

    # should these steps tell them to just immediately tag it with the final user-specified repo?
    # for now just stick with the assumption that `make release` will eventually tag a hyperkube image on gcr.io
    # and then the existing code can re-tag that for the user's repo and then push
    if ! docker inspect "${user_image_tag}" ; then
        if ! docker inspect "${official_image_tag}" ; then
            REGISTRY="gcr.io/google_containers" \
            VERSION="${KUBE_GIT_VERSION}" \
            make -C "${KUBE_ROOT}/cluster/images/hyperkube" build
        fi

        docker tag "${official_image_tag}" "${user_image_tag}"
    fi

    docker push "${user_image_tag}"

    echo "${image}:${version} was pushed to ${repo}"
    export AZURE_HYPERKUBE_SPEC="${user_image_tag}"
}

function deploy-kube-system() {
    kubectl create -f - <<EOF
apiVersion: v1
kind: Namespace
metadata:
    name: kube-system
EOF
}

function get-common-params() {
    declare -ag AZKUBE_AUTH_PARAMS
    declare -ag AZKUBE_DOCKER_PARAMS
    declare -ag AZKUBE_RESOURCE_GROUP_PARAM

    case "${AZURE_AUTH_METHOD}" in
        "client_secret")
            AZKUBE_AUTH_PARAMS+=("--client-id=${AZURE_CLIENT_ID}" "--client-secret=${AZURE_CLIENT_SECRET}")
            ;;
        "device")
            AZKUBE_AUTH_PARAMS=()
            ;;
    esac


    if [[ ! -z "${AZURE_HTTPS_PROXY:-}" ]]; then
        AZKUBE_DOCKER_PARAMS+=("--net=host" "--env=https_proxy=${AZURE_HTTPS_PROXY}")
    fi

    if [[ ! -z "${AZURE_RESOURCE_GROUP:-}" ]]; then
        echo "Forcing use of resource group ${AZURE_RESOURCE_GROUP}"
        AZKUBE_RESOURCE_GROUP_PARAM+=("--resource-group=${AZURE_RESOURCE_GROUP}")
    fi
}

function azure-deploy(){
    get-common-params

    docker run -it \
        --user "$(id -u)" \
        "${AZKUBE_DOCKER_PARAMS[@]:+${AZKUBE_DOCKER_PARAMS[@]}}" \
        -v "$HOME/.azkube:/.azkube" -v "/tmp:/tmp" \
        -v "${AZURE_OUTPUT_DIR}:/opt/azkube/${AZURE_OUTPUT_RELDIR}" \
        "colemickens/azkube:${AZKUBE_VERSION}" /opt/azkube/azkube deploy \
            --kubernetes-hyperkube-spec="${AZURE_HYPERKUBE_SPEC}" \
            --deployment-name="${AZURE_DEPLOY_ID}" \
            --location="${AZURE_LOCATION}" \
            "${AZKUBE_RESOURCE_GROUP_PARAM[@]:+${AZKUBE_RESOURCE_GROUP_PARAM[@]}}" \
            --subscription-id="${AZURE_SUBSCRIPTION_ID}" \
            --auth-method="${AZURE_AUTH_METHOD}" "${AZKUBE_AUTH_PARAMS[@]:+${AZKUBE_AUTH_PARAMS[@]}}" \
            --master-size="${AZURE_MASTER_SIZE}" \
            --node-size="${AZURE_NODE_SIZE}" \
            --node-count="${NUM_NODES}" \
            --username="${AZURE_USERNAME}" \
            --output-directory="/opt/azkube/${AZURE_OUTPUT_RELDIR}" \
            --no-cloud-provider \
            "${AZURE_AZKUBE_ARGS[@]:+${AZURE_AZKUBE_ARGS[@]}}"
}

function kube-up {
    date_start="$(date)"
    startdate="$(date +%s)"
    echo "++> AZURE KUBE-UP STARTED: $(date)"

    verify-prereqs
    azure-ensure-config

    if [[ -z "${AZURE_HYPERKUBE_SPEC:-}" ]]; then
        find-release-version
        export KUBE_GIT_VERSION="${KUBE_GIT_VERSION//+/-}"

        # this will export AZURE_HYPERKUBE_SPEC based on whether an official image was found
        # or if it was uploaded to the user specified docker repository.
        if ! ensure-hyperkube; then
            echo "Failed to ensure hyperkube was available. Exitting."
            return 1
        fi
    else
        echo "Using user specified AZURE_HYPERKUBE_SPEC: ${AZURE_HYPERKUBE_SPEC}"
        echo "Note: The existence of this is not verified! (It might only be pullable from your DC)"
    fi

    azure-deploy

    kubectl config set-cluster "${AZURE_DEPLOY_ID}" --server="https://${AZURE_DEPLOY_ID}.${AZURE_LOCATION}.cloudapp.azure.com:6443" --certificate-authority="${AZURE_OUTPUT_DIR}/ca.crt" --api-version="v1"
    kubectl config set-credentials "${AZURE_DEPLOY_ID}_user" --client-certificate="${AZURE_OUTPUT_DIR}/client.crt" --client-key="${AZURE_OUTPUT_DIR}/client.key"
    kubectl config set-context "${AZURE_DEPLOY_ID}" --cluster="${AZURE_DEPLOY_ID}" --user="${AZURE_DEPLOY_ID}_user"
    kubectl config use-context "${AZURE_DEPLOY_ID}"

    deploy-kube-system

    enddate="$(date +%s)"
    duration="$(( (startdate - enddate) ))"

    echo "++> AZURE KUBE-UP FINISHED: $(date) (duration: ${duration} seconds)"
}

function kube-down {
    verify-prereqs

    # required
    if [[ -z "${AZURE_SUBSCRIPTION_ID:-}" ]]; then
        echo "AZURE_SUBSCRIPTION_ID must be set"
        exit 1
    fi
    if [[ -z "${AZURE_DEPLOY_ID:-}" ]]; then
        echo "AZURE_DEPLOY_ID must be set. This selects the deployment (and resource group) to delete."
        return -1
    fi

    #optional
    declare -a destroy_params
    declare -a docker_params
    if [[ ${AZURE_DOWN_SKIP_CONFIRM:-} == "true" ]]; then
        destroy_params+=("--skip-confirm")
    fi
    if [[ ! -z "${AZURE_HTTPS_PROXY:-}" ]]; then
        docker_params+=("--net=host" "--env=https_proxy=${AZURE_HTTPS_PROXY}")
    fi

    docker run -it \
        --user "$(id -u)" \
        -v "$HOME/.azkube:/.azkube" -v "/tmp:/tmp" \
        "${AZKUBE_DOCKER_PARAMS[@]:+${AZKUBE_DOCKER_PARAMS[@]}}" \
        "colemickens/azkube:${AZKUBE_VERSION}" /opt/azkube/azkube destroy \
            --deployment-name="${AZURE_DEPLOY_ID}" \
            --subscription-id="${AZURE_SUBSCRIPTION_ID}" \
            --auth-method="${AZURE_AUTH_METHOD}" "${AZKUBE_AUTH_PARAMS[@]:+${AZKUBE_AUTH_PARAMS[@]}}" \
            "${destroy_params[@]:+${destroy_params[@]}}" \
            "${AZURE_AZKUBE_ARGS[@]:+${AZURE_AZKUBE_ARGS[@]}}"
}

