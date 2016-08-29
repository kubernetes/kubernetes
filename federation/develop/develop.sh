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
# the value from KUBE_VERSION.
# e.g. run as:
# KUBE_REGISTRY=localhost:5000/anushku \
# KUBE_VERSION=1.3.0-dev ./build.sh build_image
#
# will build the Docker images with the specified repository name and the
# image version tag.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT="$(dirname "${BASH_SOURCE}")/../.."
DEPLOY_ROOT="${KUBE_ROOT}/federation/deploy"
CUR_ROOT="$(pwd)"

source "${KUBE_ROOT}/build/common.sh"
source "${KUBE_ROOT}/build/util.sh"
# Provides the detect-project function
source "${KUBE_ROOT}/cluster/kube-util.sh"
# Provides logging facilities
source "${KUBE_ROOT}/cluster/lib/logging.sh"

readonly TMP_DIR="$(mktemp -d)"
readonly FEDERATION_OUTPUT_ROOT="${LOCAL_OUTPUT_ROOT}/federation"
readonly VERSIONS_FILE="${FEDERATION_OUTPUT_ROOT}/versions"

detect-project
readonly KUBE_PROJECT="${KUBE_PROJECT:-${PROJECT:-}}"

readonly KUBE_REGISTRY="${KUBE_REGISTRY:-gcr.io/${KUBE_PROJECT}}"
# In dev environments this value must be recomputed after build. See
# the build_image() function. So not making it readonly
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

function build_binaries() {
  cd "${KUBE_ROOT}"
  kube::build::verify_prereqs
  kube::build::build_image
  kube::build::run_build_command make WHAT="cmd/kubectl cmd/hyperkube"
}

function build_image() {
  # Recompute KUBE_VERSION because it might have changed after rebuild.
  local kube_version=""
  if [[ -n "${KUBE_VERSION:-}" ]]; then
    kube_version="${KUBE_VERSION}"
  else
    kube_version="$(kube::release::semantic_image_tag_version)"
    # Also append the dirty tree SHA to keep the versions unique across
    # builds.
    if [[ "${kube_version}" == *-dirty ]]; then
      kube_version+=".$(dirty_sha)"
    fi
  fi

  # Write the generated version to the output versions file so that we can
  # reuse it.
  mkdir -p "${FEDERATION_OUTPUT_ROOT}"
  jq -n --arg ver "${kube_version}" \
    '{"KUBE_VERSION": $ver}' > "${VERSIONS_FILE}"
  kube::log::status "Wrote to version file ${VERSIONS_FILE}: ${kube_version}"

  BASEIMAGE="ubuntu:16.04" \
    REGISTRY="${KUBE_REGISTRY}" \
    VERSION="${kube_version}" \
    make -C "${KUBE_ROOT}/cluster/images/hyperkube" build
}

function push() {
  local kube_version=""
  if [[ -n "${KUBE_VERSION:-}" ]]; then
    kube_version="${KUBE_VERSION}"
  else
    # Read the version back from the versions file if no version is given.
    kube_version="$(jq -r '.KUBE_VERSION' ${VERSIONS_FILE})"
  fi

  kube::log::status "Pushing hyperkube image to the registry"
  gcloud docker push "${KUBE_REGISTRY}/hyperkube-amd64:${kube_version}"

  # Update config after build and push, but before turning up the clusters
  # to ensure the config has the right image version tags.
  gen_or_update_config "${kube_version}"
}

function gen_or_update_config() {
  local -r kube_version="${1:-}"

  mkdir -p "${FEDERATION_OUTPUT_ROOT}"
  cp "${DEPLOY_ROOT}/config.json.sample" "${FEDERATION_OUTPUT_ROOT}/config.json"

  update_config \
    '[.[] | .phase1.gce.project |= "'"${KUBE_PROJECT}"'"]' \
        "${FEDERATION_OUTPUT_ROOT}/config.json"

  # Not chaining for readability
  update_config \
    '[.[] | .phase2 = { docker_registry: "'"${KUBE_REGISTRY}"'", kubernetes_version: "'"${kube_version}"'" } ]' \
    "${FEDERATION_OUTPUT_ROOT}/config.json"

  cat <<EOF> "${FEDERATION_OUTPUT_ROOT}/values.yaml"
apiserverRegistry: "${KUBE_REGISTRY}"
apiserverVersion: "${kube_version}"
controllerManagerRegistry: "${KUBE_REGISTRY}"
controllerManagerVersion: "${kube_version}"
EOF
}

readonly ACTION="${1:-}"
case "${ACTION}" in
  "")
  echo 'Action must be one of [init, build_binaries, build_image, push, \
    deploy_clusters, deploy_federation, destroy_federation, destroy_clusters \
    redeploy_federation], \
    got: '"${ACTION}"
  exit 1
  ;;
  "build_binaries")
  build_binaries
  ;;
  "build_image")
  build_image
  ;;
  "push")
  push
  ;;
  # Following functions belong to deploy.sh, they are driven from here
  # convenience during development because FEDERATION_OUTPUT_ROOT is
  # already defined during development here in this script. Also, we
  # execute the following commands in their own subshells to avoid them
  # messing with variables in this script.
  "init")
  (
    "${DEPLOY_ROOT}/deploy.sh" init
  )
  ;;
  "deploy_clusters")
  (
    export FEDERATION_OUTPUT_ROOT
    "${DEPLOY_ROOT}/deploy.sh" deploy_clusters
  )
  ;;
  "deploy_federation")
  (
    export FEDERATION_OUTPUT_ROOT
    "${DEPLOY_ROOT}/deploy.sh" deploy_federation
  )
  ;;
  "destroy_federation")
  (
    export FEDERATION_OUTPUT_ROOT
    "${DEPLOY_ROOT}/deploy.sh" destroy_federation
  )
  ;;
  "destroy_clusters")
  (
    export FEDERATION_OUTPUT_ROOT
    "${DEPLOY_ROOT}/deploy.sh" destroy_clusters
  )
  ;;
  "redeploy_federation")
  (
    export FEDERATION_OUTPUT_ROOT
    "${DEPLOY_ROOT}/deploy.sh" redeploy_federation
  )
  ;;
esac
