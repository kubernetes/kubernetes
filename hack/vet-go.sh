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


set -o errexit
set -o nounset
set -o pipefail

KUBE_REPO_ROOT=$(dirname "${BASH_SOURCE}")/..

# Set the environment variables required by the build.
source "${KUBE_REPO_ROOT}/hack/config-go.sh"

# Go to the top of the tree.
cd "${KUBE_REPO_ROOT}"

# Check for `go` binary and set ${GOPATH}.
kube::setup_go_environment

# Use eval to preserve embedded quoted strings.
eval "goflags=(${GOFLAGS:-})"

# Filter out arguments that start with "-" and move them to goflags.
targets=()
for arg; do
  if [[ "${arg}" == -* ]]; then
    goflags+=("${arg}")
  else
    targets+=("${arg}")
  fi
done

if [[ "${targets[@]+set}" != "set" ]]; then
  targets=("...")
fi

rc=0
# Filter silly "exit status 1" lines and send main output to stdout.
# This is tricky - pipefail means any non-zero exit in a pipeline is reported,
# and errexit exits on error.  Turning that into an || expression blocks the
# errexit.  But $? is still not useful because grep will return an error when it
# receives no input, which is exactly what go vet produces on success.  In short,
# if go vet fails (produces output), grep will succeed, but if go vet succeeds
# (produces no output) grep will fail.  Then we just look at PIPESTATUS[0] which
# is go's exit code.
go vet "${goflags[@]:+${goflags[@]}}" "${targets[@]/#/./}" 2>&1 \
    | grep -v "^exit status " \
    || rc=${PIPESTATUS[0]}
exit "${rc}"
