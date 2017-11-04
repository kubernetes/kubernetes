#!/bin/bash

# Copyright 2015 The Kubernetes Authors.
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

# Generates `types_swagger_doc_generated.go` files for API group
# versions. That file contains functions on API structs that return
# the comments that should be surfaced for the corresponding API type
# in our API docs.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"
source "${KUBE_ROOT}/hack/lib/swagger.sh"

kube::golang::setup_env

GROUP_VERSIONS=(meta/v1 meta/v1alpha1 ${KUBE_AVAILABLE_GROUP_VERSIONS})

# To avoid compile errors, remove the currently existing files.
for group_version in "${GROUP_VERSIONS[@]}"; do
  rm -f "$(kube::util::group-version-to-pkg-path "${group_version}")/types_swagger_doc_generated.go"
done
for group_version in "${GROUP_VERSIONS[@]}"; do
  kube::swagger::gen_types_swagger_doc "${group_version}" "$(kube::util::group-version-to-pkg-path "${group_version}")"
done
