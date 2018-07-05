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

# Runs tests for kubectl alpha diff
run_kubectl_diff_tests() {
    set -o nounset
    set -o errexit

    create_and_use_new_namespace
    kube::log::status "Testing kubectl alpha diff"

    kubectl apply -f hack/testdata/pod.yaml

    # Ensure that selfLink has been added, and shown in the diff
    output_message=$(kubectl alpha diff -f hack/testdata/pod.yaml)
    kube::test::if_has_string "${output_message}" 'selfLink'
    output_message=$(kubectl alpha diff LOCAL LIVE -f hack/testdata/pod.yaml)
    kube::test::if_has_string "${output_message}" 'selfLink'
    output_message=$(kubectl alpha diff LOCAL MERGED -f hack/testdata/pod.yaml)
    kube::test::if_has_string "${output_message}" 'selfLink'

    output_message=$(kubectl alpha diff MERGED MERGED -f hack/testdata/pod.yaml)
    kube::test::if_empty_string "${output_message}"
    output_message=$(kubectl alpha diff LIVE LIVE -f hack/testdata/pod.yaml)
    kube::test::if_empty_string "${output_message}"
    output_message=$(kubectl alpha diff LAST LAST -f hack/testdata/pod.yaml)
    kube::test::if_empty_string "${output_message}"
    output_message=$(kubectl alpha diff LOCAL LOCAL -f hack/testdata/pod.yaml)
    kube::test::if_empty_string "${output_message}"

    kubectl delete -f  hack/testdata/pod.yaml

    set +o nounset
    set +o errexit
}
