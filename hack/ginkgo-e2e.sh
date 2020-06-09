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

# This script runs e2e tests on Google Cloud Platform.
# Usage: `hack/ginkgo-e2e.sh`.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/cluster/common.sh"
source "${KUBE_ROOT}/hack/lib/init.sh"

# Find the ginkgo binary build as part of the release.
ginkgo=$(kube::util::find-binary "ginkgo")
e2e_test=$(kube::util::find-binary "e2e.test")

# --- Setup some env vars.

GINKGO_PARALLEL=${GINKGO_PARALLEL:-n} # set to 'y' to run tests in parallel
CLOUD_CONFIG=${CLOUD_CONFIG:-""}

# If 'y', Ginkgo's reporter will not print out in color when tests are run
# in parallel
GINKGO_NO_COLOR=${GINKGO_NO_COLOR:-n}

# If 'y', will rerun failed tests once to give them a second chance.
GINKGO_TOLERATE_FLAKES=${GINKGO_TOLERATE_FLAKES:-n}

: "${KUBECTL:="${KUBE_ROOT}/cluster/kubectl.sh"}"
: "${KUBE_CONFIG_FILE:="config-test.sh"}"

export KUBECTL KUBE_CONFIG_FILE

source "${KUBE_ROOT}/cluster/kube-util.sh"

function detect-master-from-kubeconfig() {
    export KUBECONFIG=${KUBECONFIG:-$DEFAULT_KUBECONFIG}

    local cc
    cc=$("${KUBE_ROOT}/cluster/kubectl.sh" config view -o jsonpath="{.current-context}")
    if [[ -n "${KUBE_CONTEXT:-}" ]]; then
      cc="${KUBE_CONTEXT}"
    fi
    local cluster
    cluster=$("${KUBE_ROOT}/cluster/kubectl.sh" config view -o jsonpath="{.contexts[?(@.name == \"${cc}\")].context.cluster}")
    KUBE_MASTER_URL=$("${KUBE_ROOT}/cluster/kubectl.sh" config view -o jsonpath="{.clusters[?(@.name == \"${cluster}\")].cluster.server}")
}

# ---- Do cloud-provider-specific setup
if [[ -n "${KUBERNETES_CONFORMANCE_TEST:-}" ]]; then
    echo "Conformance test: not doing test setup."
    KUBERNETES_PROVIDER=${KUBERNETES_CONFORMANCE_PROVIDER:-"skeleton"}

    detect-master-from-kubeconfig

    auth_config=(
      "--kubeconfig=${KUBECONFIG}"
    )
else
    echo "Setting up for KUBERNETES_PROVIDER=\"${KUBERNETES_PROVIDER}\"."

    prepare-e2e

    detect-master >/dev/null

    KUBE_MASTER_URL="${KUBE_MASTER_URL:-}"
    if [[ -z "${KUBE_MASTER_URL:-}" && -n "${KUBE_MASTER_IP:-}" ]]; then
      KUBE_MASTER_URL="https://${KUBE_MASTER_IP}"
    fi

    auth_config=(
      "--kubeconfig=${KUBECONFIG:-$DEFAULT_KUBECONFIG}"
    )
fi

if [[ -n "${NODE_INSTANCE_PREFIX:-}" ]]; then
  NODE_INSTANCE_GROUP="${NODE_INSTANCE_PREFIX}-group"
fi

if [[ "${KUBERNETES_PROVIDER}" == "gce" ]]; then
  set_num_migs
  NODE_INSTANCE_GROUP=""
  for ((i=1; i<=NUM_MIGS; i++)); do
    if [[ ${i} == "${NUM_MIGS}" ]]; then
      # We are assigning the same mig names as create-nodes function from cluster/gce/util.sh.
      NODE_INSTANCE_GROUP="${NODE_INSTANCE_GROUP}${NODE_INSTANCE_PREFIX}-group"
    else
      NODE_INSTANCE_GROUP="${NODE_INSTANCE_GROUP}${NODE_INSTANCE_PREFIX}-group-${i},"
    fi
  done
fi

# TODO(kubernetes/test-infra#3330): Allow NODE_INSTANCE_GROUP to be
# set before we get here, which eliminates any cluster/gke use if
# KUBERNETES_CONFORMANCE_PROVIDER is set to "gke".
if [[ -z "${NODE_INSTANCE_GROUP:-}" ]] && [[ "${KUBERNETES_PROVIDER}" == "gke" ]]; then
  detect-node-instance-groups
  NODE_INSTANCE_GROUP=$(kube::util::join , "${NODE_INSTANCE_GROUP[@]}")
fi

if [[ "${KUBERNETES_PROVIDER}" == "azure" ]]; then
    if [[ ${CLOUD_CONFIG} == "" ]]; then
        echo "Missing azure cloud config"
        exit 1
    fi
fi

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

# The --host setting is used only when providing --auth_config
# If --kubeconfig is used, the host to use is retrieved from the .kubeconfig
# file and the one provided with --host is ignored.
# Add path for things like running kubectl binary. 
PATH=$(dirname "${e2e_test}"):"${PATH}"
export PATH
"${ginkgo}" "${ginkgo_args[@]:+${ginkgo_args[@]}}" "${e2e_test}" -- \
  "${auth_config[@]:+${auth_config[@]}}" \
  --ginkgo.flakeAttempts="${FLAKE_ATTEMPTS}" \
  --host="${KUBE_MASTER_URL}" \
  --provider="${KUBERNETES_PROVIDER}" \
  --gce-project="${PROJECT:-}" \
  --gce-zone="${ZONE:-}" \
  --gce-region="${REGION:-}" \
  --gce-multizone="${MULTIZONE:-false}" \
  --gke-cluster="${CLUSTER_NAME:-}" \
  --kube-master="${KUBE_MASTER:-}" \
  --cluster-tag="${CLUSTER_ID:-}" \
  --cloud-config-file="${CLOUD_CONFIG:-}" \
  --repo-root="${KUBE_ROOT}" \
  --node-instance-group="${NODE_INSTANCE_GROUP:-}" \
  --prefix="${KUBE_GCE_INSTANCE_PREFIX:-e2e}" \
  --network="${KUBE_GCE_NETWORK:-${KUBE_GKE_NETWORK:-e2e}}" \
  --node-tag="${NODE_TAG:-}" \
  --master-tag="${MASTER_TAG:-}" \
  --docker-config-file="${DOCKER_CONFIG_FILE:-}" \
  --dns-domain="${KUBE_DNS_DOMAIN:-cluster.local}" \
  --ginkgo.slowSpecThreshold="${GINKGO_SLOW_SPEC_THRESHOLD:-300}" \
  ${KUBE_CONTAINER_RUNTIME:+"--container-runtime=${KUBE_CONTAINER_RUNTIME}"} \
  ${MASTER_OS_DISTRIBUTION:+"--master-os-distro=${MASTER_OS_DISTRIBUTION}"} \
  ${NODE_OS_DISTRIBUTION:+"--node-os-distro=${NODE_OS_DISTRIBUTION}"} \
  ${NUM_NODES:+"--num-nodes=${NUM_NODES}"} \
  ${E2E_REPORT_DIR:+"--report-dir=${E2E_REPORT_DIR}"} \
  ${E2E_REPORT_PREFIX:+"--report-prefix=${E2E_REPORT_PREFIX}"} \
  "${@:-}"
