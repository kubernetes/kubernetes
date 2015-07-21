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

# List cherry picks sitting on a release branch that AREN'T YET part
# of a release. Used when constructing a release.

set -o errexit
set -o nounset
set -o pipefail

declare -r KUBE_ROOT="$(dirname "${BASH_SOURCE}")/.."

if [[ "$#" -ne 1 ]]; then
  echo "${0} <remote branch>: list all automated cherry picks on <remote release branch> since last release."
  echo "  (assumes branch format of cherry_pick_pull.sh)"
  echo ""
  echo "Example:"
  echo "  $0 upstream/release-1.0  # Lists all PRs on release-1.0 since list patch release."
  exit 2
fi

declare -r BRANCH="$1"

git remote update >/dev/null

# First, the range specification: --abbrev=0 is saying to find the tag
# relevant for the branch, so this essentially the git log all the way
# back to the most recent tag on the release branch.
#
# The git log outputs something like:
#  0f3cdb7234e2239707e4c3fc58f5f89552f41c65 Merge pull request #98765 from zaphod/automated-cherry-pick-of-#12345-#56789-#13579-upstream-release-1.0
PULLS=( $(git log $(git describe --abbrev=0 "${BRANCH}").."${BRANCH}" -E --grep="Merge pull request \#[0-9]+ from .+/automated-cherry-pick-of-" --pretty=oneline |
  awk '{ print $7 }' | sed -e 's/.*automated-cherry-pick-of-\(#[0-9]\{1,\}\)/\1/' -e 's/\(#[0-9]\{1,\}\)-[^#].*/\1/' -e 's/-/\
/g' | sed 's/^#//g') )

${KUBE_ROOT}/hack/lookup_pull.py "${PULLS[@]}"
