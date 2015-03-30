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

# Build a LMKTFY release.  This will build the binaries, create the Docker
# images and other build artifacts.  All intermediate artifacts will be hosted
# publicly on Google Cloud Storage currently.

set -o errexit
set -o nounset
set -o pipefail

LMKTFY_ROOT=$(dirname "${BASH_SOURCE}")/..
source "$LMKTFY_ROOT/build/common.sh"

LMKTFY_RELEASE_RUN_TESTS=${LMKTFY_RELEASE_RUN_TESTS-y}

lmktfy::build::verify_prereqs
lmktfy::build::build_image
lmktfy::build::run_build_command hack/build-cross.sh

if [[ $LMKTFY_RELEASE_RUN_TESTS =~ ^[yY]$ ]]; then
  lmktfy::build::run_build_command hack/test-go.sh
  lmktfy::build::run_build_command hack/test-integration.sh
fi

lmktfy::build::copy_output
lmktfy::release::package_tarballs
lmktfy::release::gcs::release
