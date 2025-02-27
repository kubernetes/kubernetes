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
# 4. Create kind cluster using kubetest2 and run e2e tests.
# 5. Collect logs and metrics from kind cluster.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../../../.. && pwd -P)"
source "${KUBE_ROOT}/hack/lib/init.sh"

readonly cluster_name="kms"
readonly registry_name="kind-registry"
readonly kind_network="kind"

# build_and_push_mock_plugin builds and pushes the kms mock plugin image to the local registry.
build_and_push_mock_plugin() {
    docker buildx build \
        --no-cache \
        --platform linux/amd64 \
        --output=type=docker \
        --build-arg=GOTOOLCHAIN="${GOTOOLCHAIN}" \
        -t localhost:5000/mock-kms-provider:e2e \
        -f staging/src/k8s.io/kms/internal/plugins/_mock/Dockerfile staging/src/k8s.io/ \
        --progress=plain;

    docker push localhost:5000/mock-kms-provider:e2e
}

# create_registry creates local registry if not already present.
create_registry() {
    running="$(docker inspect -f '{{.State.Running}}' "${registry_name}" 2>/dev/null || true)"
    if [ "${running}" != 'true' ]; then
        echo "Creating local registry"
        docker run \
            -d --restart=always -p "5000:5000" --name "${registry_name}" \
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
        if docker network ls | grep "${kind_network}"; then
            break
        else
            echo "'docker network ls' does not have '${kind_network}' network yet. Retrying in 1 second..."
            sleep 1
    fi
    done

    containers=$(docker network inspect "${kind_network}" -f "{{range .Containers}}{{.Name}} {{end}}")
    needs_connect="true"
    for c in $containers; do
        if [ "$c" = "${registry_name}" ]; then
            needs_connect="false"
        fi
    done

    if [ "${needs_connect}" = "true" ]; then
        echo "connecting kind network to local registry"
        docker network connect "${kind_network}" "${registry_name}"
    else
        echo "'${kind_network}' network is already connected to local registry"
    fi
}

# create_cluster_and_run_test creates a kind cluster using kubetest2 and runs e2e tests.
create_cluster_and_run_test() {
    CLUSTER_CREATE_ATTEMPTED=true

    TEST_ARGS=""
    if [ "${SKIP_RUN_TESTS:-}" != "true" ]; then
        # (--use-built-binaries) use the kubectl, e2e.test, and ginkgo binaries built during --build as opposed to from a GCS release tarball
        TEST_ARGS="--test=ginkgo -- --focus-regex=\[Conformance\] --skip-regex=\[Serial\] --parallel 20 --use-built-binaries"
    else
        echo "Skipping running tests"
    fi

    # shellcheck disable=SC2086
    kubetest2 kind -v 5 \
    --build \
    --up \
    --rundir-in-artifacts \
    --config test/e2e/testing-manifests/auth/encrypt/kind.yaml \
    --cluster-name "${cluster_name}" ${TEST_ARGS}
}

cleanup() {
    # CLUSTER_CREATE_ATTEMPTED is true once we run kubetest2 kind --up
    if [ "${CLUSTER_CREATE_ATTEMPTED:-}" = true ]; then
        if [ "${SKIP_COLLECT_LOGS:-}" != "true" ]; then
            # collect logs and metrics
            echo "Collecting logs"
            mkdir -p "${ARTIFACTS}/logs"
            kind "export" logs "${ARTIFACTS}/logs" --name "${cluster_name}"

            echo "Collecting metrics"
            mkdir -p "${ARTIFACTS}/metrics"
            kubectl get --raw /metrics > "${ARTIFACTS}/metrics/kube-apiserver-metrics.txt"
        else
            echo "Skipping collecting logs and metrics"
        fi

        if [ "${SKIP_DELETE_CLUSTER:-}" != "true" ]; then
            echo "Deleting kind cluster"
            # delete cluster
            kind delete cluster --name "${cluster_name}"
        else
            echo "Skipping deleting kind cluster"
        fi
    fi
}

main(){
    # ensure artifacts (results) directory exists when not in CI
    export ARTIFACTS="${ARTIFACTS:-${PWD}/_artifacts}"
    mkdir -p "${ARTIFACTS}"

    kube::golang::setup_env
    (
        # just while installing external tools
        export GO111MODULE=on GOTOOLCHAIN=auto
        # TODO: consider using specific versions to avoid surprise breaking changes
        go install sigs.k8s.io/kind@latest
        go install sigs.k8s.io/kubetest2@latest
        go install sigs.k8s.io/kubetest2/kubetest2-kind@latest
        go install sigs.k8s.io/kubetest2/kubetest2-tester-ginkgo@latest
    )

    # The build e2e.test, ginkgo and kubectl binaries + copy to dockerized dir is
    # because of https://github.com/kubernetes-sigs/kubetest2/issues/184
    make all WHAT="test/e2e/e2e.test vendor/github.com/onsi/ginkgo/v2/ginkgo cmd/kubectl"
    mkdir -p _output/dockerized/bin/linux/amd64;
    for binary in kubectl e2e.test ginkgo; do
        cp -f _output/local/go/bin/${binary} _output/dockerized/bin/linux/amd64/${binary}
    done;

    create_registry
    build_and_push_mock_plugin
    connect_registry &
    create_cluster_and_run_test
    cleanup
}

trap cleanup INT TERM
main "$@"
