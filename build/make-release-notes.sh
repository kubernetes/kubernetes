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

# Clean out the output directory on the docker host.
set -o errexit
set -o nounset
set -o pipefail

function pop_dir {
  popd > /dev/null
}

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

if [[ -z "${1:-}" ]]; then
  echo "Usage: ${0} <last-release-pr-number> <current-release-pr-number> --api-token=$TOKEN [opts]"
  echo "To create a GitHub API token, see https://github.com/settings/tokens"
  exit 1
fi

pushd . > /dev/null
trap 'pop_dir' INT TERM EXIT

kube::golang::build_binaries contrib/release-notes
kube::golang::place_bins
echo "Fetching release notes"
releasenotes=$(kube::util::find-binary "release-notes")
"${releasenotes}" --last-release-pr=${1} --current-release-pr=${2} ${@}

