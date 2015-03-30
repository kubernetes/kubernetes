#!/bin/bash

# Copyright 2014 Google Inc. All rights reserved.
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
LMKTFY_GCS_NO_CACHING=n
LMKTFY_GCS_MAKE_PUBLIC=y
LMKTFY_GCS_UPLOAD_RELEASE=y
LMKTFY_GCS_RELEASE_BUCKET=lmktfy-release
LMKTFY_GCS_PROJECT=google-containers
LMKTFY_GCS_RELEASE_PREFIX="ci/${LATEST}"
LMKTFY_GCS_LATEST_FILE="ci/latest.txt"
LMKTFY_GCS_LATEST_CONTENTS=${LATEST}

LMKTFY_ROOT=$(dirname "${BASH_SOURCE}")/..
source "$LMKTFY_ROOT/build/common.sh"

lmktfy::release::gcs::release
lmktfy::release::gcs::publish_latest
