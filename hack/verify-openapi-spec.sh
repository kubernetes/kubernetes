#!/usr/bin/env bash

# Copyright 2016 The Kubernetes Authors.
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

# This script checks whether updating of OpenAPI specification is needed or not.
# It verifies that the OpenAPI specification is up to date in strict mode, and
# will fallback to check in non-strict mode if that fails. Strict mode removes
# all APIs marked # as removed in a particular version, while non-strict mode
# allows them to persist until the release cutoff. We allow non-strict to
# prevent CI failures when we bump the version number in the git tag.
# We should run `hack/update-openapi-spec.sh` if OpenAPI specification is out of
# date.
# Usage: `hack/verify-openapi-spec.sh`.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..

source "${KUBE_ROOT}/hack/lib/verify-generated.sh"
(
  kube::verify::generated "Generated files failed strict alpha check and MAY need be updated" "Running verification again without strict alpha check" hack/update-openapi-spec.sh "$@"
) || \
KUBE_APISERVER_STRICT_REMOVED_API_HANDLING_IN_ALPHA=false kube::verify::generated "Generated files need to be updated" "Please run 'hack/update-openapi-spec.sh'" hack/update-openapi-spec.sh "$@"