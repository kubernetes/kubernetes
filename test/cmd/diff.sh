#!/usr/bin/env bash

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

set -o errexit
set -o nounset
set -o pipefail

# Runs tests for kubectl diff
run_kubectl_diff_tests() {
    set -o nounset
    set -o errexit

    create_and_use_new_namespace
    kube::log::status "Testing kubectl diff"

    # Test that it works when the live object doesn't exist
    output_message=$(! kubectl diff -f hack/testdata/pod.yaml)
    kube::test::if_has_string "${output_message}" 'test-pod'
    # Ensure diff only dry-runs and doesn't persist change
    kube::test::get_object_assert 'pod' "{{range.items}}{{ if eq ${id_field:?} \\\"test-pod\\\" }}found{{end}}{{end}}:" ':'

    kubectl apply -f hack/testdata/pod.yaml
    kube::test::get_object_assert 'pod' "{{range.items}}{{ if eq ${id_field:?} \\\"test-pod\\\" }}found{{end}}{{end}}:" 'found:'
    initialResourceVersion=$(kubectl get "${kube_flags[@]:?}" -f hack/testdata/pod.yaml -o go-template='{{ .metadata.resourceVersion }}')

    # Make sure that diffing the resource right after returns nothing (0 exit code).
    kubectl diff -f hack/testdata/pod.yaml

    # Ensure diff only dry-runs and doesn't persist change
    resourceVersion=$(kubectl get "${kube_flags[@]:?}" -f hack/testdata/pod.yaml -o go-template='{{ .metadata.resourceVersion }}')
    kube::test::if_has_string "${resourceVersion}" "${initialResourceVersion}"

    # Make sure that:
    # 1. the exit code for diff is 1 because it found a difference
    # 2. the difference contains the changed image
    output_message=$(kubectl diff -f hack/testdata/pod-changed.yaml || test $? -eq 1)
    kube::test::if_has_string "${output_message}" 'k8s.gcr.io/pause:3.0'

    # Ensure diff only dry-runs and doesn't persist change
    resourceVersion=$(kubectl get "${kube_flags[@]:?}" -f hack/testdata/pod.yaml -o go-template='{{ .metadata.resourceVersion }}')
    kube::test::if_has_string "${resourceVersion}" "${initialResourceVersion}"

    # Test found diff with server-side apply
    output_message=$(kubectl diff -f hack/testdata/pod-changed.yaml --server-side --force-conflicts || test $? -eq 1)
    kube::test::if_has_string "${output_message}" 'k8s.gcr.io/pause:3.0'

    # Ensure diff --server-side only dry-runs and doesn't persist change
    resourceVersion=$(kubectl get "${kube_flags[@]:?}" -f hack/testdata/pod.yaml -o go-template='{{ .metadata.resourceVersion }}')
    kube::test::if_has_string "${resourceVersion}" "${initialResourceVersion}"

    # Test that we have a return code bigger than 1 if there is an error when diffing
    kubectl diff -f hack/testdata/invalid-pod.yaml || test $? -gt 1

    # Cleanup
    kubectl delete -f hack/testdata/pod.yaml

    kube::log::status "Testing kubectl diff with server-side apply"

    # Test that kubectl diff --server-side works when the live object doesn't exist
    output_message=$(! kubectl diff --server-side -f hack/testdata/pod.yaml)
    kube::test::if_has_string "${output_message}" 'test-pod'
    # Ensure diff --server-side only dry-runs and doesn't persist change
    kube::test::get_object_assert 'pod' "{{range.items}}{{ if eq ${id_field:?} \\\"test-pod\\\" }}found{{end}}{{end}}:" ':'

    # Server-side apply the Pod
    kubectl apply --server-side -f hack/testdata/pod.yaml
    kube::test::get_object_assert 'pod' "{{range.items}}{{ if eq ${id_field:?} \\\"test-pod\\\" }}found{{end}}{{end}}:" 'found:'

    # Make sure that --server-side diffing the resource right after returns nothing (0 exit code).
    kubectl diff --server-side -f hack/testdata/pod.yaml

    # Make sure that for kubectl diff --server-side:
    # 1. the exit code for diff is 1 because it found a difference
    # 2. the difference contains the changed image
    output_message=$(kubectl diff --server-side -f hack/testdata/pod-changed.yaml || test $? -eq 1)
    kube::test::if_has_string "${output_message}" 'k8s.gcr.io/pause:3.0'

    # Cleanup
    kubectl delete -f hack/testdata/pod.yaml

    set +o nounset
    set +o errexit
}

run_kubectl_diff_same_names() {
    set -o nounset
    set -o errexit

    create_and_use_new_namespace
    kube::log::status "Test kubectl diff with multiple resources with the same name"

    output_message=$(KUBECTL_EXTERNAL_DIFF="find" kubectl diff -Rf hack/testdata/diff/)
    kube::test::if_has_string "${output_message}" 'v1\.Pod\..*\.test'
    kube::test::if_has_string "${output_message}" 'apps\.v1\.Deployment\..*\.test'
    kube::test::if_has_string "${output_message}" 'v1\.ConfigMap\..*\.test'
    kube::test::if_has_string "${output_message}" 'v1\.Secret\..*\.test'

    set +o nounset
    set +o errexit
}
