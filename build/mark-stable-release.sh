#!/bin/bash

# Copyright 2015 The Kubernetes Authors All rights reserved.
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

# Marks the current stable version

set -o errexit
set -o nounset
set -o pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: ${0} <version>"
  exit 1
fi

KUBE_RELEASE_VERSION="${1-}"

KUBE_GCS_MAKE_PUBLIC='y'
KUBE_GCS_RELEASE_BUCKET='kubernetes-release'
KUBE_GCS_RELEASE_PREFIX="release/${KUBE_RELEASE_VERSION}"
KUBE_GCS_PUBLISH_VERSION="${KUBE_RELEASE_VERSION}"

KUBE_ROOT="$(dirname "${BASH_SOURCE}")/.."
source "${KUBE_ROOT}/build/common.sh"

if "${KUBE_ROOT}/cluster/kubectl.sh" 'version' | grep 'Client' | grep 'dirty'; then
  echo "!!! Tag at invalid point, or something else is bad. Build is dirty. Don't push this build." >&2
  exit 1
fi

kube::release::gcs::publish_official 'stable'
