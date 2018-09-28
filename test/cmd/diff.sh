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
    output_message=$(kubectl diff -f hack/testdata/pod.yaml)
    kube::test::if_has_string "${output_message}" 'test-pod'

    kubectl apply -f hack/testdata/pod.yaml

    output_message=$(kubectl diff -f hack/testdata/pod-changed.yaml)
    kube::test::if_has_string "${output_message}" 'k8s.gcr.io/pause:3.0'

    kubectl delete -f  hack/testdata/pod.yaml

    set +o nounset
    set +o errexit
}
