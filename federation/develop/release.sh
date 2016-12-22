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

set -o errexit
set -o nounset
set -o pipefail

readonly FEDERATION_OUTPUT_ROOT="${LOCAL_OUTPUT_ROOT}/federation"
readonly VERSIONS_FILE="${FEDERATION_OUTPUT_ROOT}/versions"

readonly KUBE_PROJECT="${KUBE_PROJECT:-${PROJECT:-}}"
readonly KUBE_REGISTRY="${KUBE_REGISTRY:-gcr.io/${KUBE_PROJECT}}"

function get_version() {
  local kube_version=""
  # $KUBERNETES_RELEASE is set in the CI builds and that must be used.
  # $VERSIONS_FILE is only for local builds.
  if [[ -n "${KUBERNETES_RELEASE:-}" ]]; then
    kube_version="${KUBERNETES_RELEASE}"
  else
    # Read the version back from the versions file if no version is given.
    kube_version="$(cat ${VERSIONS_FILE} | python -c '\
import json, sys;\
print json.load(sys.stdin)["KUBE_VERSION"]')"
  fi
  echo "${kube_version}"
}
