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

: ${KUBE_VERSION_ROOT:=${KUBE_ROOT}}
: ${KUBECTL:="${KUBE_VERSION_ROOT}/cluster/kubectl.sh"}
: ${KUBE_CONFIG_FILE:="config-test.sh"}

export KUBECTL KUBE_CONFIG_FILE

source "${KUBE_ROOT}/cluster/kube-env.sh"
source "${KUBE_VERSION_ROOT}/cluster/${KUBERNETES_PROVIDER}/util.sh"

prepare-e2e

detect-master >/dev/null

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

if [[ "$KUBERNETES_PROVIDER" == "vagrant" ]]; then
  # When we are using vagrant it has hard coded auth.  We repeat that here so that
  # we don't clobber auth that might be used for a publicly facing cluster.
  auth_config=(
    "--auth_config=$HOME/.kubernetes_vagrant_auth"
  )
elif [[ "${KUBERNETES_PROVIDER}" == "gke" ]]; then
  # With GKE, our auth and certs are in gcloud's config directory.
  detect-project &> /dev/null
  cfg_dir="${GCLOUD_CONFIG_DIR}/${PROJECT}.${ZONE}.${CLUSTER_NAME}"
  auth_config=(
    "--auth_config=${cfg_dir}/kubernetes_auth"
    "--cert_dir=${cfg_dir}"
  )
elif [[ "${KUBERNETES_PROVIDER}" == "gce" ]]; then
  auth_config=(
    "--auth_config=${HOME}/.kube/${INSTANCE_PREFIX}/kubernetes_auth"
  )
else
  auth_config=()
fi

"${e2e}" "${auth_config[@]:+${auth_config[@]}}" \
  --host="https://${KUBE_MASTER_IP-}" \
  --provider="${KUBERNETES_PROVIDER}" \
  ${E2E_REPORT_DIR+"--report_dir=${E2E_REPORT_DIR}"} \
  "${@:-}"
