#!/usr/bin/env bash

# Copyright 2016 The Kubernetes Authors.
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

# This script will build the hyperkube image and push it to the repository
# referred to by KUBE_REGISTRY. The image will be given a version tag with
# the value from KUBE_VERSION. It also turns up/turns down Kubernetes
# clusters and federation components using the built hyperkube image.
# e.g. run as:
# KUBE_REGISTRY=localhost:5000/anushku \
# KUBE_VERSION=1.3.0-dev ./build.sh
#
# will deploy the components using
# localhost:5000/anushku/hyperkube-amd64:1.3.0-dev image.

# Everything in this script is expected to be executed from the $KUBE_ROOT
# directory.

# TODO(madhusudancs): Separate the dev functions from the deployment
# functions. A lot of code here is to make this work in dev environments.
# The script that we ship to the users as part of a release should be
# much simpler (about 80% of the code here could be removed for non-dev
# environments).

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
CUR_ROOT=$(dirname "${BASH_SOURCE}")

source "${KUBE_ROOT}/build/common.sh"
source "${KUBE_ROOT}/build/util.sh"
# Provides the $KUBERNETES_PROVIDER variable and detect-project function
source "${KUBE_ROOT}/cluster/kube-util.sh"
source "${KUBE_ROOT}/cluster/lib/logging.sh"

readonly ACTION="${1:-gen}"

readonly TMP_DIR="$(mktemp -d)"
readonly FEDERATION_OUTPUT_ROOT="${LOCAL_OUTPUT_ROOT}/federation"

readonly KUBE_ANYWHERE_FEDERATION_IMAGE="gcr.io/madhusudancs-containers/kubernetes-anywhere-federation"
readonly KUBE_ANYWHERE_FEDERATION_VERSION="v0.9.0"
readonly KUBE_ANYWHERE_FEDERATION_CHARTS_IMAGE="gcr.io/madhusudancs-containers/federation-charts"
readonly KUBE_ANYWHERE_FEDERATION_CHARTS_VERSION="v0.9.0"

readonly GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-${HOME}/.config/gcloud/application_default_credentials.json}"
readonly KUBE_CONFIG_DIR="${KUBE_CONFIG_DIR:-${HOME}/.kube}"
readonly KUBE_CONFIG="${KUBE_CONFIG:-${HOME}/.kube/config}"

detect-project
readonly KUBE_PROJECT="${KUBE_PROJECT:-${PROJECT:-}}"

readonly KUBE_REGISTRY="${KUBE_REGISTRY:-gcr.io/${KUBE_PROJECT}}"
# In dev environments this value must be recomputed after build. See
# the build() function. Not making it readonly
KUBE_VERSION="${KUBE_VERSION:-}"


function cleanup {
  rm -rf "${TMP_DIR}"
  cd "${CUR_ROOT}"
}
trap cleanup EXIT

function dirty_sha() {
  local -r index="${KUBE_ROOT}/.git/index"
  local -r objects_dir="${KUBE_ROOT}/.git/objects"

  local -r tmp_dir="${TMP_DIR}/.git"
  local -r tmp_index="${tmp_dir}/index"
  local -r tmp_objects_dir="${tmp_dir}/objects"

  mkdir -p "${tmp_objects_dir}"
  cp "${index}" "${tmp_index}"

  local -r files=$(git ls-files -m -o -d --exclude-standard)
  GIT_INDEX_FILE="${tmp_index}" git add ${files}
  GIT_ALTERNATE_OBJECT_DIRECTORIES="${objects_dir}" GIT_OBJECT_DIRECTORY="${tmp_objects_dir}" GIT_INDEX_FILE="${tmp_index}" git write-tree
}

function update_config() {
  local -r q="${1:-}"
  local -r cfile="${2:-}"
  local -r bname="$(basename ${cfile})"

  jq "${q}" "${cfile}" > "${TMP_DIR}/${bname}"
  mv "${TMP_DIR}/${bname}" "${cfile}"
}

function build() {
  kube::build::verify_prereqs
  kube::build::build_image
  kube::build::run_build_command make WHAT="cmd/kubectl cmd/hyperkube"

  # Recompute KUBE_VERSION because it might have changed after rebuild.
  KUBE_VERSION="${KUBE_VERSION:-$(kube::release::semantic_image_tag_version)}"

  # Also append the dirty tree SHA to keep the versions unique across
  # builds.
  if [[ "${KUBE_VERSION}" == *-dirty ]]; then
    KUBE_VERSION+=".$(dirty_sha)"
  fi

  BASEIMAGE="ubuntu:16.04" \
    REGISTRY="${KUBE_REGISTRY}" \
    VERSION="${KUBE_VERSION}" \
    make -C "${KUBE_ROOT}/cluster/images/hyperkube" build
}

function push() {
  kube::log::status "Pushing hyperkube image to the registry"
  gcloud docker push "${KUBE_REGISTRY}/hyperkube-amd64:${KUBE_VERSION}"
}

function pull_installer() {
  kube::log::status "Pulling installer images"
  docker pull "${KUBE_ANYWHERE_FEDERATION_IMAGE}:${KUBE_ANYWHERE_FEDERATION_VERSION}"
  docker pull "${KUBE_ANYWHERE_FEDERATION_CHARTS_IMAGE}:${KUBE_ANYWHERE_FEDERATION_CHARTS_VERSION}"
}

function ensure_files() {
  kube::log::status "Ensure provider is supported..."
  if [[ "${KUBERNETES_PROVIDER:-}" != "gce" ]]; then
    echo "Supported providers: \"gce\""
    exit 1
  fi

  kube::log::status "Ensure credential files exist..."
  if [[ ! -f "${GOOGLE_APPLICATION_CREDENTIALS}" ]]; then
    echo "Please ensure Google credentials file \""${GOOGLE_APPLICATION_CREDENTIALS}"\" exists."
    exit 1
  fi

  if [[ ! -f "${KUBE_CONFIG}" ]]; then
    echo "Please ensure kubeconfig file \""${KUBE_CONFIG}"\" exists."
    exit 1
  fi
}

function kube_action() {
  kube::log::status "${ACTION} clusters"
  docker run \
    --user="$(id -u):$(id -g)" \
    -m 12G \
    -v "${GOOGLE_APPLICATION_CREDENTIALS}:/.config/gcloud/application_default_credentials.json:ro" \
    -v "${KUBE_CONFIG_DIR}:/.kube" \
    -v "${FEDERATION_OUTPUT_ROOT}:/_output" \
    "${KUBE_ANYWHERE_FEDERATION_IMAGE}:${KUBE_ANYWHERE_FEDERATION_VERSION}" \
    "${ACTION}"
}

function federation_action() {
  kube::log::status "${ACTION} federation components"
  docker run \
    -m 12G \
    -v "${KUBE_CONFIG}:/root/.kube/config:ro" \
    -v "${FEDERATION_OUTPUT_ROOT}:/_output" \
    "${KUBE_ANYWHERE_FEDERATION_CHARTS_IMAGE}:${KUBE_ANYWHERE_FEDERATION_CHARTS_VERSION}" \
    "${ACTION}"
}

function gen_or_update_config() {
  mkdir -p "${FEDERATION_OUTPUT_ROOT}"
  cp "federation/config.default.json" "${FEDERATION_OUTPUT_ROOT}/config.json"

  update_config \
    '[.[] | .phase1.gce.project |= "'"${KUBE_PROJECT}"'"]' \
        "${FEDERATION_OUTPUT_ROOT}/config.json"

  # Not chaining for readability
  update_config \
    '[.[] | .phase2 = { docker_registry: "'"${KUBE_REGISTRY}"'", kubernetes_version: "'"${KUBE_VERSION}"'" } ]' \
    "${FEDERATION_OUTPUT_ROOT}/config.json"

  cat <<EOF> "${FEDERATION_OUTPUT_ROOT}/values.yaml"
apiserverRegistry: "${KUBE_REGISTRY}"
apiserverVersion: "${KUBE_VERSION}"
controllerManagerRegistry: "${KUBE_REGISTRY}"
controllerManagerVersion: "${KUBE_VERSION}"
EOF
}

if [[ "${ACTION}" == "gen" || "${ACTION}" == "deploy" ]]; then
  ensure_files

  cd "${KUBE_ROOT}"
  build
  push

  pull_installer

  # Update config after build and push, but before turning up the clusters
  # to ensure the config has the right image version tags.
  gen_or_update_config

  kube_action
  federation_action
else
  federation_action
  kube_action
fi

kube::log::status "Successfully completed!"
