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

# Script generates test case for test/integration/apiserver/compatibility_versions_test.go - TestFeatureGateCompatibilityEmulationVersion.
# Usage: `test/integration/apiserver/generate-compatibility-versions-test.sh` 1.31.1 kindest/node:v1.31.1

set -o errexit
set -o nounset
set -o pipefail

# Ensure the previous version and kind image are passed as arguments
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <PREV_VERSION> <KIND_IMAGE_FOR_PREV_VERSION> (eg: v1.31.1 kindest/node:v1.31.1)"
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
  kind create cluster --name "${PREV_VERSION}" --image "${2}"
fi

echo "==========Test Case for ${PREV_VERSION}=============="
echo "emulationVersion: ${PREV_VERSION},"

# Create temporary files
metrics_file=$(mktemp)
sources_file=$(mktemp)

# Get metrics data
kubectl get --raw /metrics | grep kubernetes_feature_enabled > "$metrics_file"

# Get source code
git show "${PREV_VERSION}":staging/src/k8s.io/apiserver/pkg/features/kube_features.go \
  "${PREV_VERSION}":pkg/features/versioned_kube_features.go \
  "${PREV_VERSION}":staging/src/k8s.io/apiextensions-apiserver/pkg/features/kube_features.go > "$sources_file"

# Process both files through the combined parser
go run hack/tools/parse-feature-gates/parse-feature-gates.go "$metrics_file" "$sources_file"

# Clean up
rm "$metrics_file" "$sources_file"

echo "====================================================="
kind delete cluster --name "${PREV_VERSION}"