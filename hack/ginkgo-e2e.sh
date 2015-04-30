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

GINKGO_PARALLEL=${GINKGO_PARALLEL:-n} # set to 'y' to run tests in parallel
KUBE_ROOT=$(readlink -f $(dirname "${BASH_SOURCE}")/..)

source "${KUBE_ROOT}/cluster/common.sh"
source "${KUBE_ROOT}/hack/lib/init.sh"

# Ginkgo will build the e2e tests, so we need to make sure that the environment
# is set up correctly (including Godeps, etc).
kube::golang::setup_env

# --- Build the Ginkgo test runner from Godeps
godep go install github.com/onsi/ginkgo/ginkgo
ginkgo=$(godep path)/bin/ginkgo

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

    if [[ "${KUBERNETES_PROVIDER}" == "gke" ]]; then
      # GKE stores its own kubeconfig in gcloud's config directory.
      detect-project &> /dev/null
      auth_config=(
        "--kubeconfig=${GCLOUD_CONFIG_DIR}/kubeconfig"
        # gcloud doesn't set the current-context, so we have to set it
        "--context=gke_${PROJECT}_${ZONE}_${CLUSTER_NAME}"
      )
    elif [[ "${KUBERNETES_PROVIDER}" == "conformance_test" ]]; then
      auth_config=(
        "--auth-config=${KUBERNETES_CONFORMANCE_TEST_AUTH_CONFIG:-}"
        "--cert-dir=${KUBERNETES_CONFORMANCE_TEST_CERT_DIR:-}"
      )
    else
      auth_config=(
      "--kubeconfig=${KUBECONFIG:-$DEFAULT_KUBECONFIG}"
    )
    fi
else
  echo "Conformance Test.  No cloud-provider-specific preparation."
  KUBERNETES_PROVIDER=""
  auth_config=(
    "--auth-config=${AUTH_CONFIG:-}"
    "--cert-dir=${CERT_DIR:-}"
  )
fi

ginkgo_args=()
if [[ ${GINKGO_PARALLEL} =~ ^[yY]$ ]]; then
  ginkgo_args+=("-p")
fi

# The --host setting is used only when providing --auth_config
# If --kubeconfig is used, the host to use is retrieved from the .kubeconfig
# file and the one provided with --host is ignored.
"${ginkgo}" "${ginkgo_args[@]:+${ginkgo_args[@]}}" test/e2e -- \
  "${auth_config[@]:+${auth_config[@]}}" \
  --host="https://${KUBE_MASTER_IP-}" \
  --provider="${KUBERNETES_PROVIDER}" \
  --gce-project="${PROJECT:-}" \
  --gce-zone="${ZONE:-}" \
  --kube-master="${KUBE_MASTER:-}" \
  --repo-root="${KUBE_VERSION_ROOT}" \
  ${E2E_REPORT_DIR+"--report-dir=${E2E_REPORT_DIR}"} \
  "${@:-}"
