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

run_kubectl_auth_whoami_tests() {
    set -o nounset
    set -o errexit

    kube::log::status "Testing kubectl auth whoami"

    # Command
    output_message=$(kubectl auth whoami -o json 2>&1)

    # Post-condition: should return user attributes.
    kube::test::if_has_string "${output_message}" '"kind": "SelfSubjectReview"'

    set +o nounset
    set +o errexit
}
