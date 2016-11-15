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

# This script turns up/turns down Kubernetes clusters and federation
# components using the built hyperkube image.
# e.g. run as:
# FEDERATION_OUTPUT_ROOT="./_output" ./deploy.sh deploy_clusters
#
# will deploy the kubernetes clusters using the configuration specified
# in $FEDERATION_OUTPUT_ROOT/config.json.
#
# See config.json.sample for a config.json example.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..

# Provides the $KUBERNETES_PROVIDER variable and detect-project function
source "${KUBE_ROOT}/cluster/kube-util.sh"
# Provides logging facilities
source "${KUBE_ROOT}/cluster/lib/logging.sh"

readonly KUBE_ANYWHERE_FEDERATION_IMAGE="gcr.io/madhusudancs-containers/kubernetes-anywhere-federation"
readonly KUBE_ANYWHERE_FEDERATION_VERSION="v0.9.0"
readonly KUBE_ANYWHERE_FEDERATION_CHARTS_IMAGE="gcr.io/madhusudancs-containers/federation-charts"
readonly KUBE_ANYWHERE_FEDERATION_CHARTS_VERSION="v0.9.1"

readonly GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-${HOME}/.config/gcloud/application_default_credentials.json}"
readonly KUBE_CONFIG_DIR="${KUBE_CONFIG_DIR:-${HOME}/.kube}"
readonly KUBE_CONFIG="${KUBE_CONFIG:-${HOME}/.kube/config}"

function pull_installer() {
  kube::log::status "Pulling installer images"
  docker pull "${KUBE_ANYWHERE_FEDERATION_IMAGE}:${KUBE_ANYWHERE_FEDERATION_VERSION}"
  docker pull "${KUBE_ANYWHERE_FEDERATION_CHARTS_IMAGE}:${KUBE_ANYWHERE_FEDERATION_CHARTS_VERSION}"
}

function ensure_files() {
  kube::log::status "Ensure provider is supported"
  if [[ "${KUBERNETES_PROVIDER:-}" != "gce" && "${KUBERNETES_PROVIDER:-}" != "gke" ]]; then
    echo "Supported providers: \"gce\", \"gke\""
    exit 1
  fi

  kube::log::status "Ensure credential files exist"
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
  : "${FEDERATION_OUTPUT_ROOT:?must be set}"

  local -r action="${1:-}"
  kube::log::status "Action: ${action} clusters"
  docker run \
    --user="$(id -u):$(id -g)" \
    -m 12G \
    -v "${GOOGLE_APPLICATION_CREDENTIALS}:/.config/gcloud/application_default_credentials.json:ro" \
    -v "${KUBE_CONFIG_DIR}:/.kube" \
    -v "${FEDERATION_OUTPUT_ROOT}:/_output" \
    "${KUBE_ANYWHERE_FEDERATION_IMAGE}:${KUBE_ANYWHERE_FEDERATION_VERSION}" \
    "${action}"
}

function federation_action() {
  : "${FEDERATION_OUTPUT_ROOT:?must be set}"

  local -r action="${1:-}"
  kube::log::status "Action: ${action} federation components"
  # For non-GKE clusters just mounting kubeconfig is sufficient. But we
  # need gcloud credentials for GKE clusters, so we pass both kubeconfig
  # and gcloud credentials
  docker run \
    -m 12G \
    -v "${GOOGLE_APPLICATION_CREDENTIALS}:/root/.config/gcloud/application_default_credentials.json:ro" \
    -v "${KUBE_CONFIG}:/root/.kube/config" \
    -v "${FEDERATION_OUTPUT_ROOT}:/_output" \
    "${KUBE_ANYWHERE_FEDERATION_CHARTS_IMAGE}:${KUBE_ANYWHERE_FEDERATION_CHARTS_VERSION}" \
    "${action}"
}

function redeploy_federation() {
  : "${FEDERATION_OUTPUT_ROOT:?must be set}"

  local -r action="${1:-}"
  kube::log::status "${action} federation components"
  docker run \
    -m 12G \
    -v "${KUBE_CONFIG}:/root/.kube/config:ro" \
    -v "${FEDERATION_OUTPUT_ROOT}:/_output" \
    "${KUBE_ANYWHERE_FEDERATION_CHARTS_IMAGE}:${KUBE_ANYWHERE_FEDERATION_CHARTS_VERSION}" \
    "${action}"
}

readonly ACTION="${1:-}"
case "${ACTION}" in
  "")
  echo 'Action must be one of [init, deploy_clusters, deploy_federation, \
    destroy_federation, destroy_clusters, redeploy_federation], \
    got: '"${ACTION}"
  exit 1
  ;;
  "init")
  pull_installer
  ;;
  "deploy_clusters")
  ensure_files
  kube_action deploy
  ;;
  "deploy_federation")
  ensure_files
  federation_action deploy
  ;;
  "destroy_federation")
  federation_action destroy
  ;;
  "destroy_clusters")
  kube_action destroy
  ;;
  "redeploy_federation")
  redeploy_federation
  ;;
esac
