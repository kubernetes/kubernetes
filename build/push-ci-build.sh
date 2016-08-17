#!/bin/bash

# Copyright 2014 The Kubernetes Authors.
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

# Pushes a continuous integration build to our official CI repository

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..

source "${KUBE_ROOT}/build/util.sh"

LATEST=$(kube::release::semantic_version)

KUBE_GCS_NO_CACHING='n'
KUBE_GCS_MAKE_PUBLIC='y'
KUBE_GCS_UPLOAD_RELEASE='y'
KUBE_GCS_DELETE_EXISTING='n'
: ${KUBE_GCS_RELEASE_BUCKET:='kubernetes-release-dev'}
KUBE_GCS_RELEASE_PREFIX="ci/${LATEST}"
KUBE_GCS_PUBLISH_VERSION="${LATEST}"
: ${KUBE_GCS_UPDATE_LATEST:='y'}

source "${KUBE_ROOT}/build/common.sh"

MAX_ATTEMPTS=3
attempt=0
while [[ ${attempt} -lt ${MAX_ATTEMPTS} ]]; do
  if [[ "${KUBE_GCS_UPDATE_LATEST}" =~ ^[yY]$ ]]; then
    kube::release::gcs::release && kube::release::gcs::publish_ci && break || true
  else
    kube::release::gcs::release && break || true
  fi
  attempt=$((attempt + 1))
  sleep 5
done
if [[ ! ${attempt} -lt ${MAX_ATTEMPTS} ]];then
    kube::log::error "Max attempts reached. Will exit."
    exit 1
fi

if [[ "${FEDERATION:-}" == "true" ]];then
    "${KUBE_ROOT}/build/push-federation-images.sh"
fi
