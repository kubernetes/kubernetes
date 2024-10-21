#!/usr/bin/env bash

# Copyright 2024 The Kubernetes Authors.
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

# This script generates a test case for the test/integration/apiserver/apiserver_test.go - TestFeatureGateCompatibilityEmulationVersion test.
# Usage: `hack/update-integration-apiserver-test-feature-gate-compatibility-emulation-version.sh` 1.31.1

set -o errexit
set -o nounset
set -o pipefail

# Ensure the previous version is passed as an argument
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <PREV_VERSION> (eg: v1.31.1)"
  exit 1
fi

PREV_VERSION="$1"

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::golang::setup_env

# Check if cluster with name $PREV_VERSION already exists
if kind get clusters | grep -q "^${PREV_VERSION}$"; then
  echo "Cluster ${PREV_VERSION} already exists, skipping creation."
else
  # Create a kind cluster for the previous versions if a cluster doesn't already exist
  echo "Creating kind cluster ${PREV_VERSION}..."
  kind create cluster --name "${PREV_VERSION}" --image kindest/node:"${PREV_VERSION}"
fi

echo "==========Test Case for ${PREV_VERSION}=============="
# Output the emulationVersion with the version passed in as an argument
echo "emulationVersion: ${PREV_VERSION},"
# Get the raw /metrics values from the cluster, parse them, and write to stdout the enabled feature gates for the previous version cluster
kubectl get --raw /metrics | grep kubernetes_feature_enabled | go run hack/tools/parse-feature-gates/parse-feature-gates.go
# Parse the feature gates that have "LockToDefault: true" from source files and write them to stdout
# NOTE: below requires that the local kubernetes/kubernetes git repository has N-1..3 branches and/or tags pulled down
git show "${PREV_VERSION}":staging/src/k8s.io/apiserver/pkg/features/kube_features.go \
  "${PREV_VERSION}":pkg/features/versioned_kube_features.go \
  "${PREV_VERSION}":staging/src/k8s.io/apiextensions-apiserver/pkg/features/kube_features.go \
  | go run hack/tools/parse-lock-to-default/parse-lock-to-default.go
echo "====================================================="

kind delete cluster --name "${PREV_VERSION}"
