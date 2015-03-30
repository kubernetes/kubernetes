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

# Run a command in the docker build container.  Typically this will be one of
# the commands in `hack/`.  When running in the build container the user is sure
# to have a consistent reproducible build environment.

set -o errexit
set -o nounset
set -o pipefail

LMKTFY_ROOT=$(dirname "${BASH_SOURCE}")/..
source "$LMKTFY_ROOT/build/common.sh"

lmktfy::build::verify_prereqs
lmktfy::build::build_image
lmktfy::build::run_build_command "$@"

if [[ ${LMKTFY_RUN_COPY_OUTPUT:-y} =~ ^[yY]$ ]]; then
  lmktfy::build::copy_output
fi
