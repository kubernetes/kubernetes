#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

# input and default values
PLUGIN_NAME="${1:-base64}"
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
echo "Building ${PLUGIN_NAME} image"
make ${PLUGIN_NAME}-docker-build
echo "Pushing image to local registry"
docker push "${REGISTRY}/${PLUGIN_NAME}:e2e"
