#!/usr/bin/env bash

# Copyright 2023 The Kubernetes Authors.
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

# input and default values
REGISTRY_NAME="${2:-kind-registry}"
REGISTRY_PORT="${3:-5000}"
ARCH="${4:-amd64}"

# create registry container unless it already exists
running="$(docker inspect -f '{{.State.Running}}' "${REGISTRY_NAME}" 2>/dev/null || true)"
if [ "${running}" != 'true' ]; then
  echo "Creating local registry"
  docker run \
    -d --restart=always -p "${REGISTRY_PORT}:5000" --name "${REGISTRY_NAME}" \
    registry:2
fi

# Build and push kms image
export REGISTRY=localhost:${REGISTRY_PORT}
export OUTPUT_TYPE=type=docker

# push build image to local registry
echo "Build and push image to local registry"
make base64-docker-build
echo "Pushing image to local registry"
docker push "${REGISTRY}/base64:e2e"
