#!/usr/bin/env bash

# Copyright The Kubernetes Authors.
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

# This script runs to ensure that packages which have opted into
# strict Go API change tracking have an up-to-date CHANGELOG.md.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/util.sh"

dirs=()
kube::util::read-array dirs < <(grep -v -e'^#' -e '^$' "${KUBE_ROOT}/hack/go-apidocs.conf")

# Base and target revisions are detected automatically by the script.
if "${KUBE_ROOT}/hack/apidiff.sh" -u "${dirs[@]}"; then
    cat <<EOF

Congratulations, no Go API changes need to be documented in CHANGELOG.md files!
EOF
else
    cat <<EOF

One or more CHANGELOG.md files were updated using placeholder text.
Replace that text and include the documentation of your change in
your branch.
EOF
fi
