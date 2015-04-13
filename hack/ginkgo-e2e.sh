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

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..

# --- Find local test binaries.
source "${KUBE_ROOT}/hack/lib/util.sh"
e2e=$(kube::util::find-binary "e2e")

# --- Setup some env vars.

: ${KUBE_VERSION_ROOT:=${KUBE_ROOT}}
: ${KUBECTL:="${KUBE_VERSION_ROOT}/cluster/kubectl.sh"}
: ${KUBE_CONFIG_FILE:="config-test.sh"}

export KUBECTL KUBE_CONFIG_FILE

source "${KUBE_ROOT}/cluster/kube-env.sh"

# ---- Do cloud-provider-specific setup
if [[ -z "${AUTH_CONFIG:-}" ]];  then
    echo "Setting up for KUBERNETES_PROVIDER=\"${KUBERNETES_PROVIDER}\"."

    source "${KUBE_VERSION_ROOT}/cluster/${KUBERNETES_PROVIDER}/util.sh"

    prepare-e2e

    detect-master >/dev/null


    if [[ "$KUBERNETES_PROVIDER" == "vagrant" ]]; then
      # When we are using vagrant it has hard coded auth.  We repeat that here so that
      # we don't clobber auth that might be used for a publicly facing cluster.
      auth_config=(
        "--auth_config=${HOME}/.kubernetes_vagrant_auth"
        "--kubeconfig=${HOME}/.kubernetes_vagrant_kubeconfig"
      )
    elif [[ "${KUBERNETES_PROVIDER}" == "gke" ]]; then
      # GKE stores its own kubeconfig in gcloud's config directory.
      detect-project &> /dev/null
      auth_config=(
        "--kubeconfig=${GCLOUD_CONFIG_DIR}/kubeconfig"
        # gcloud doesn't set the current-context, so we have to set it
        "--context=gke_${PROJECT}_${ZONE}_${CLUSTER_NAME}"
      )
    elif [[ "${KUBERNETES_PROVIDER}" == "gce" ]]; then
      auth_config=(
        "--kubeconfig=${HOME}/.kube/.kubeconfig"
      )
    elif [[ "${KUBERNETES_PROVIDER}" == "aws" ]]; then
      auth_config=(
        "--auth_config=${HOME}/.kube/${INSTANCE_PREFIX}/kubernetes_auth"
      )
    elif [[ "${KUBERNETES_PROVIDER}" == "libvirt-coreos" ]]; then
      auth_config=(
        "--kubeconfig=${HOME}/.kube/.kubeconfig"
      )
    elif [[ "${KUBERNETES_PROVIDER}" == "conformance_test" ]]; then
      auth_config=(
        "--auth_config=${KUBERNETES_CONFORMANCE_TEST_AUTH_CONFIG:-}"
        "--cert_dir=${KUBERNETES_CONFORMANCE_TEST_CERT_DIR:-}"
      )
    else
      auth_config=()
    fi
else
  echo "Conformance Test.  No cloud-provider-specific preparation."
  KUBERNETES_PROVIDER=""
  auth_config=(
    "--auth_config=${AUTH_CONFIG:-}"
    "--cert_dir=${CERT_DIR:-}"
  )
fi

# Use the kubectl binary from the same directory as the e2e binary.
# The --host setting is used only when providing --auth_config
# If --kubeconfig is used, the host to use is retrieved from the .kubeconfig
# file and the one provided with --host is ignored.
export PATH=$(dirname "${e2e}"):"${PATH}"
"${e2e}" "${auth_config[@]:+${auth_config[@]}}" \
  --host="https://${KUBE_MASTER_IP-}" \
  --provider="${KUBERNETES_PROVIDER}" \
  --gce_project="${PROJECT:-}" \
  --gce_zone="${ZONE:-}" \
  --kube_master="${KUBE_MASTER:-}" \
  ${E2E_REPORT_DIR+"--report_dir=${E2E_REPORT_DIR}"} \
  "${@:-}"
