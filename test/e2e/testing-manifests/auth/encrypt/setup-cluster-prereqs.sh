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

# build mock plugin
build_and_push_mock_plugin() {
    docker buildx build \
        --no-cache \
        --platform linux/amd64 \
        --output=type=docker \
        -t localhost:5000/base64:e2e \
        -f staging/src/k8s.io/kms/internal/plugins/mock/Dockerfile staging/src/k8s.io/ \
        --progress=plain;

    docker push localhost:5000/base64:e2e
}

# create local registry
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

# connect registry to kind network
connect_registry(){
    # wait for the kind network to exist
    for ((; ;)); do
        if docker network ls | grep "kind"; then
            break
        else
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
        docker network connect "kind" "kind-registry" || true
    else
        echo "'kind' network is already connected to 'kind-registry'"
    fi
}

# create cluster and run tests using kubetest2
create_cluster() {
    echo "creating cluster and running tests"
    kubetest2 kind -v 5 \
    --build \
    --up \
    --rundir /home/anramase/go/src/k8s.io/kubernetes/_rundir \
    --config test/e2e/testing-manifests/auth/encrypt/kind.yaml \
    --cluster-name kms \
    --test=ginkgo \
    -- \
    --v=5 \
    --focus-regex='\[Conformance\]' \
    --skip-regex='\[Serial\]' \
    --parallel 20 \
    --use-built-binaries
}

collect_metrics() {
    echo "collecting metrics"
    mkdir -p "${ARTIFACTS}/metrics"
    kubectl get --raw /metrics > "${ARTIFACTS}/metrics/kube-apiserver-metrics.txt"
}

main() {
    # ensure artifacts (results) directory exists when not in CI
    export ARTIFACTS="${ARTIFACTS:-${PWD}/_artifacts}"
    mkdir -p "${ARTIFACTS}"

    create_registry
    build_and_push_mock_plugin
    connect_registry &
    create_cluster
    collect_metrics
}

main "$@"
exit 0
