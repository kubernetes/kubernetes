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

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::golang::setup_env

make -C "${KUBE_ROOT}" WHAT=cmd/hyperkube

# add other BADSYMBOLS here.
BADSYMBOLS=(
  "httptest"
  "testify"
  "testing[.]"
)

# b/c hyperkube binds everything simply check that for bad symbols
SYMBOLS="$(nm ${KUBE_OUTPUT_HOSTBIN}/hyperkube)"

RESULT=0
for BADSYMBOL in "${BADSYMBOLS[@]}"; do
  if FOUND=$(echo "$SYMBOLS" | grep "$BADSYMBOL"); then
    echo "Found bad symbol '${BADSYMBOL}':"
    echo "$FOUND"
    RESULT=1
  fi
done

exit $RESULT

# ex: ts=2 sw=2 et filetype=sh
