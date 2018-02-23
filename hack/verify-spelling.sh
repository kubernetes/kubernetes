#!/usr/bin/env bash
# Copyright 2018 The Kubernetes Authors.
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

export KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

# Install tools we need, but only from vendor/...
cd ${KUBE_ROOT}
go install ./vendor/github.com/client9/misspell/cmd/misspell
if ! which misspell >/dev/null 2>&1; then
    echo "Can't find misspell - is your GOPATH 'bin' in your PATH?" >&2
    echo "  GOPATH: ${GOPATH}" >&2
    echo "  PATH:   ${PATH}" >&2
    exit 1
fi

# Spell checking
# All the skipping files are defined in hack/.spelling_failures
skipping_file="${KUBE_ROOT}/hack/.spelling_failures"
failing_packages=$(echo `cat ${skipping_file}` | sed "s| | -e |g")
git ls-files | grep -v -e ${failing_packages} | xargs misspell -i "Creater,creater,ect" -error -o stderr
