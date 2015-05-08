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

LATEST=$(git describe)
KUBE_GCS_NO_CACHING=n
KUBE_GCS_MAKE_PUBLIC=y
KUBE_GCS_UPLOAD_RELEASE=y
KUBE_GCS_RELEASE_BUCKET=kubernetes-release
KUBE_GCS_RELEASE_PREFIX="ci/${LATEST}"
KUBE_GCS_LATEST_FILE="ci/latest.txt"
KUBE_GCS_LATEST_CONTENTS=${LATEST}

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "$KUBE_ROOT/build/common.sh"

kube::release::gcs::release
kube::release::gcs::publish_latest
