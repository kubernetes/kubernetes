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

# This script sets up a go workspace locally and builds all go components.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..

# Set the environment variables required by the build.
source "${KUBE_ROOT}/hack/config-go.sh"

# Go to the top of the tree.
cd "${KUBE_ROOT}"

# Check for `go` binary and set ${GOPATH}.
kube::setup_go_environment

# Fetch the version.
version_ldflags=$(kube::version_ldflags)

# Use eval to preserve embedded quoted strings.
eval "goflags=(${GOFLAGS:-})"

targets=()
for arg; do
  if [[ "${arg}" == -* ]]; then
    # Assume arguments starting with a dash are flags to pass to go.
    goflags+=("${arg}")
  else
    targets+=("${arg}")
  fi
done

if [[ ${#targets[@]} -eq 0 ]]; then
  targets=($(kube::default_build_targets))
fi

binaries=($(kube::binaries_from_targets "${targets[@]}"))

echo "Building local go components"
# Note that the flags to 'go build' are duplicated in the dockerized build setup
# (build/build-image/common.sh).  If we add more command line options to our
# standard build we'll want to duplicate them there.  This needs to be fixed
go install "${goflags[@]:+${goflags[@]}}" \
    -ldflags "${version_ldflags}" \
    "${binaries[@]}"
