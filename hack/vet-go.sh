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


set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::golang::setup_env

cd "${KUBE_ROOT}"

# Use eval to preserve embedded quoted strings.
eval "goflags=(${KUBE_GOFLAGS:-})"

# Filter out arguments that start with "-" and move them to goflags.
targets=()
for arg; do
  if [[ "${arg}" == -* ]]; then
    goflags+=("${arg}")
  else
    targets+=("${arg}")
  fi
done

if [[ ${#targets[@]} -eq 0 ]]; then
  targets=("...")
fi

# Filter silly "exit status 1" lines and send main output to stdout.
#
# This is tricky - pipefail means any non-zero exit in a pipeline is reported,
# and errexit exits on error.  Turning that into an || expression blocks the
# errexit.  But $? is still not useful because grep will return an error when it
# receives no input, which is exactly what go vet produces on success.  In
# short, if go vet fails (produces output), grep will succeed, but if go vet
# succeeds (produces no output) grep will fail.  Then we just look at
# PIPESTATUS[0] which is go's exit code.
rc=0
go vet "${goflags[@]:+${goflags[@]}}" "${targets[@]/#/./}" 2>&1 \
    | grep -v "^exit status " \
    || rc=${PIPESTATUS[0]}
exit "${rc}"
