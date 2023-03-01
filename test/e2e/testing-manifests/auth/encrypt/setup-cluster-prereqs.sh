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

# This script does following:
# 1. Creates local registry if not already present. This registry is used to push the kms mock plugin image.
# 2. Build and push the kms mock plugin image to the local registry.
# 3. Connect local registry to kind network so that kind cluster created using kubetest2 in prow CI job can pull the kms mock plugin image.

set -o errexit
set -o nounset
set -o pipefail

# build_and_push_mock_plugin builds and pushes the kms mock plugin image to the local registry.
build_and_push_mock_plugin() {
    docker buildx build \
        --no-cache \
        --platform linux/amd64 \
        --output=type=docker \
        -t localhost:5000/mock-kms-provider:e2e \
        -f staging/src/k8s.io/kms/internal/plugins/mock/Dockerfile staging/src/k8s.io/ \
        --progress=plain;

    docker push localhost:5000/mock-kms-provider:e2e
}

# create_registry creates local registry if not already present.
create_registry() {
    running="$(docker inspect -f '{{.State.Running}}' "kind-registry" 2>/dev/null || true)"
    if [ "${running}" != 'true' ]; then
        echo "Creating local registry"
        docker run \
            -d --restart=always -p "5000:5000" --name "kind-registry" \
            registry:2
    else 
        echo "Local registry is already running"
    fi
}

# connect_registry connects local registry to kind network.
connect_registry(){
    # wait for the kind network to exist
    # infinite loop here is fine because kubetest2 will timeout if kind cluster creation fails and that will terminate the CI job
    for ((; ;)); do
        if docker network ls | grep "kind"; then
            break
        else
            echo "'docker network ls' does not have 'kind' network to connect registry"
            sleep 1
    fi
    done

    containers=$(docker network inspect "kind" -f "{{range .Containers}}{{.Name}} {{end}}")
    needs_connect="true"
    for c in $containers; do
        if [ "$c" = "kind-registry" ]; then
            needs_connect="false"
        fi
    done

    if [ "${needs_connect}" = "true" ]; then
        echo "connecting kind network to kind-registry"
        docker network connect "kind" "kind-registry"
    else
        echo "'kind' network is already connected to 'kind-registry'"
    fi
}

main(){
    create_registry
    build_and_push_mock_plugin
    connect_registry &
}

main
