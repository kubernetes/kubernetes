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

# Sourcable temp directory functions

set -o errexit
set -o nounset
set -o pipefail
set -o errtrace

# Runs the supplied command string in a temporary workspace directory.
function cluster::mesos::docker::run_in_temp_dir {
  prefix="$1"
  shift
  cmd="$@"

  # create temp WORKSPACE dir in current dir to avoid permission issues of TMPDIR on mac os x
  local -r workspace=$(env TMPDIR=$(pwd) mktemp -d -t "${prefix}-XXXXXX")
  echo "Workspace created: ${workspace}" 1>&2

  cleanup() {
    local -r workspace="$1"
    rm -rf "${workspace}"
    echo "Workspace deleted: ${workspace}" 1>&2
  }
  trap "cleanup '${workspace}'" EXIT

  pushd "${workspace}" > /dev/null
  ${cmd}
  popd > /dev/null

  trap - EXIT
  cleanup "${workspace}"
}
