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

LMKTFY_ROOT=$(dirname "${BASH_SOURCE}")/..

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
  "${LMKTFY_ROOT}/_output/dockerized/bin/${host_os}/${host_arch}/e2e"
  "${LMKTFY_ROOT}/_output/local/bin/${host_os}/${host_arch}/e2e"
  "${LMKTFY_ROOT}/platforms/${host_os}/${host_arch}/e2e"
)
e2e=$( (ls -t "${locations[@]}" 2>/dev/null || true) | head -1 )

# --- Setup some env vars.

: ${LMKTFY_VERSION_ROOT:=${LMKTFY_ROOT}}
: ${LMKTFYCTL:="${LMKTFY_VERSION_ROOT}/cluster/lmktfyctl.sh"}
: ${LMKTFY_CONFIG_FILE:="config-test.sh"}

export LMKTFYCTL LMKTFY_CONFIG_FILE

source "${LMKTFY_ROOT}/cluster/lmktfy-env.sh"

# ---- Do cloud-provider-specific setup
if [[ -z "${AUTH_CONFIG:-}" ]];  then
    echo "Setting up for LMKTFYRNETES_PROVIDER=\"${LMKTFYRNETES_PROVIDER}\"."

    source "${LMKTFY_VERSION_ROOT}/cluster/${LMKTFYRNETES_PROVIDER}/util.sh"

    prepare-e2e

    detect-master >/dev/null


    if [[ "$LMKTFYRNETES_PROVIDER" == "vagrant" ]]; then
      # When we are using vagrant it has hard coded auth.  We repeat that here so that
      # we don't clobber auth that might be used for a publicly facing cluster.
      auth_config=(
        "--auth_config=${HOME}/.lmktfy_vagrant_auth"
        "--lmktfyconfig=${HOME}/.lmktfy_vagrant_lmktfyconfig"
      )
    elif [[ "${LMKTFYRNETES_PROVIDER}" == "gke" ]]; then
      # With GKE, our auth and certs are in gcloud's config directory.
      detect-project &> /dev/null
      cfg_dir="${GCLOUD_CONFIG_DIR}/${PROJECT}.${ZONE}.${CLUSTER_NAME}"
      auth_config=(
        "--auth_config=${cfg_dir}/lmktfy_auth"
        "--cert_dir=${cfg_dir}"
      )
    elif [[ "${LMKTFYRNETES_PROVIDER}" == "gce" ]]; then
      auth_config=(
        "--lmktfyconfig=${HOME}/.lmktfy/.lmktfyconfig"
      )
    elif [[ "${LMKTFYRNETES_PROVIDER}" == "aws" ]]; then
      auth_config=(
        "--auth_config=${HOME}/.lmktfy/${INSTANCE_PREFIX}/lmktfy_auth"
      )
    elif [[ "${LMKTFYRNETES_PROVIDER}" == "libvirt-coreos" ]]; then
      auth_config=(
        "--lmktfyconfig=${HOME}/.lmktfy/.lmktfyconfig"
      )
    elif [[ "${LMKTFYRNETES_PROVIDER}" == "conformance_test" ]]; then
      auth_config=(
        "--auth_config=${LMKTFYRNETES_CONFORMANCE_TEST_AUTH_CONFIG:-}"
        "--cert_dir=${LMKTFYRNETES_CONFORMANCE_TEST_CERT_DIR:-}"
      )
    else
      auth_config=()
    fi
else
  echo "Conformance Test.  No cloud-provider-specific preparation."
  LMKTFYRNETES_PROVIDER=""
  auth_config=(
    "--auth_config=${AUTH_CONFIG:-}"
    "--cert_dir=${CERT_DIR:-}"
  )
fi

# Use the lmktfyctl binary from the same directory as the e2e binary.
# The --host setting is used only when providing --auth_config
# If --lmktfyconfig is used, the host to use is retrieved from the .lmktfyconfig
# file and the one provided with --host is ignored.
export PATH=$(dirname "${e2e}"):"${PATH}"
"${e2e}" "${auth_config[@]:+${auth_config[@]}}" \
  --host="https://${LMKTFY_MASTER_IP-}" \
  --provider="${LMKTFYRNETES_PROVIDER}" \
  --gce_project="${PROJECT:-}" \
  --gce_zone="${ZONE:-}" \
  --lmktfy_master="${LMKTFY_MASTER:-}" \
  ${E2E_REPORT_DIR+"--report_dir=${E2E_REPORT_DIR}"} \
  "${@:-}"
