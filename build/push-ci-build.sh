#!/bin/bash

# Copyright 2014 The Kubernetes Authors All rights reserved.
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

# TODO(zmerlynn): Blech, this belongs in build/common.sh, probably,
# but common.sh sets up its readonly variables when its sourced, so
# there's a chicken/egg issue getting it there and using it for
# KUBE_GCE_RELEASE_PREFIX.
function kube::release::semantic_version() {
  # This takes:
  # Client Version: version.Info{Major:"1", Minor:"1+", GitVersion:"v1.1.0-alpha.0.2328+3c0a05de4a38e3", GitCommit:"3c0a05de4a38e355d147dbfb4d85bad6d2d73bb9", GitTreeState:"clean"}
  # and spits back the GitVersion piece in a way that is somewhat
  # resilient to the other fields changing (we hope)
  ${KUBE_ROOT}/cluster/kubectl.sh version -c | sed "s/, */\\
/g" | egrep "^GitVersion:" | cut -f2 -d: | cut -f2 -d\"
}

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
LATEST=$(kube::release::semantic_version)

KUBE_GCS_NO_CACHING='n'
KUBE_GCS_MAKE_PUBLIC='y'
KUBE_GCS_UPLOAD_RELEASE='y'
KUBE_GCS_DELETE_EXISTING='y'
KUBE_GCS_RELEASE_BUCKET='kubernetes-release'
KUBE_GCS_RELEASE_PREFIX="ci/${LATEST}"
KUBE_GCS_PUBLISH_VERSION="${LATEST}"

source "$KUBE_ROOT/build/common.sh"

MAX_ATTEMPTS=3
attempt=0
while [[ ${attempt} -lt ${MAX_ATTEMPTS} ]]; do
  kube::release::gcs::release && kube::release::gcs::publish_ci && break || true
  attempt=$((attempt + 1))
  sleep 5
done
[[ ${attempt} -lt ${MAX_ATTEMPTS} ]] || exit 1
