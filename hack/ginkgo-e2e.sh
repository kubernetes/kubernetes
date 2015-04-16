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
source "${KUBE_ROOT}/cluster/common.sh"

# --- Find local test binaries.

# Detect the OS name/arch so that we can find our binary
case "$(uname -s)" in
  Darwin)
    host_os=darwin
    ;;
  Linux)
    host_os=linux
    ;;
  *)
    echo "Unsupported host OS.  Must be Linux or Mac OS X." >&2
    exit 1
    ;;
esac

case "$(uname -m)" in
  x86_64*)
    host_arch=amd64
    ;;
  i?86_64*)
    host_arch=amd64
    ;;
  amd64*)
    host_arch=amd64
    ;;
  arm*)
    host_arch=arm
    ;;
  i?86*)
    host_arch=x86
    ;;
  *)
    echo "Unsupported host arch. Must be x86_64, 386 or arm." >&2
    exit 1
    ;;
esac

# Gather up the list of likely places and use ls to find the latest one.
locations=(
  "${KUBE_ROOT}/_output/dockerized/bin/${host_os}/${host_arch}/e2e"
  "${KUBE_ROOT}/_output/local/bin/${host_os}/${host_arch}/e2e"
  "${KUBE_ROOT}/platforms/${host_os}/${host_arch}/e2e"
)
e2e=$( (ls -t "${locations[@]}" 2>/dev/null || true) | head -1 )

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
        "--auth_config=${KUBERNETES_CONFORMANCE_TEST_AUTH_CONFIG:-}"
        "--cert_dir=${KUBERNETES_CONFORMANCE_TEST_CERT_DIR:-}"
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
