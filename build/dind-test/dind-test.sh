#!/bin/bash

# Copyright 2018 The Kubernetes Authors.
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

# This is heavily duplicated with hack/ginkgo-e2e.sh. Unfortunately,
# ginkgo-e2e.sh is highly coupled to both the k8s repo and cloud provider
# behavior. We plan to refactor this, but for now it's easiest to just copy the
# functionality we want.

set -o errexit
set -o nounset
set -o pipefail

# Find the ginkgo binary build as part of the release.
ginkgo="/ginkgo"
e2e_test="/e2e.test"

CONFORMANE_TEST_SKIP_REGEX=${CONFORMANE_TEST_SKIP_REGEX:-"(Feature:Example)|(NFS)|(StatefulSet)"}

# Configure ginkgo
GINKGO_PARALLEL=${GINKGO_PARALLEL:-y} # set to 'y' to run tests in parallel

# If 'y', Ginkgo's reporter will not print out in color when tests are run
# in parallel
GINKGO_NO_COLOR=${GINKGO_NO_COLOR:-n}

# If 'y', will rerun failed tests once to give them a second chance.
GINKGO_TOLERATE_FLAKES=${GINKGO_TOLERATE_FLAKES:-n}

ginkgo_args=()
if [[ -n "${CONFORMANCE_TEST_SKIP_REGEX:-}" ]]; then
  ginkgo_args+=("--skip=${CONFORMANCE_TEST_SKIP_REGEX}")
  ginkgo_args+=("--seed=1436380640")
fi
if [[ -n "${GINKGO_PARALLEL_NODES:-}" ]]; then
  ginkgo_args+=("--nodes=${GINKGO_PARALLEL_NODES}")
elif [[ ${GINKGO_PARALLEL} =~ ^[yY]$ ]]; then
  ginkgo_args+=("--nodes=25")
fi

if [[ "${GINKGO_UNTIL_IT_FAILS:-}" == true ]]; then
  ginkgo_args+=("--untilItFails=true")
fi

FLAKE_ATTEMPTS=1
if [[ "${GINKGO_TOLERATE_FLAKES}" == "y" ]]; then
  FLAKE_ATTEMPTS=2
fi

if [[ "${GINKGO_NO_COLOR}" == "y" ]]; then
  ginkgo_args+=("--noColor")
fi

# ---- Do cloud-provider-specific setup
KUBECONFIG="/var/kubernetes/admin.conf"
auth_config="--kubeconfig=${KUBECONFIG}"

# --- Setup some env vars.
: ${KUBECTL:="/usr/bin/kubectl"}

export KUBECTL

# The --host setting is used only when providing --auth_config
# If --kubeconfig is used, the host to use is retrieved from the .kubeconfig
# file and the one provided with --host is ignored.
# Add path for things like running kubectl binary.
export PATH=$(dirname "${e2e_test}"):"${PATH}"
"${ginkgo}" "${ginkgo_args[@]:+${ginkgo_args[@]}}" "${e2e_test}" -- \
  "${auth_config[@]:+${auth_config[@]}}" \
  --ginkgo.flakeAttempts="${FLAKE_ATTEMPTS}" \
  --num-nodes=4 \
  --systemd-services=docker,kubelet \
  --report-dir=/ \
  "${@:-}"
