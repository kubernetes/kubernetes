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

# Bumps the version number by creating a couple of commits.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..

NEW_VERSION=${1-}

fetch_url=$(git remote -v | grep GoogleCloudPlatform/kubernetes.git | grep fetch | awk '{ print $2 }')
if ! push_url=$(git remote -v | grep GoogleCloudPlatform/kubernetes.git | grep push | awk '{ print $2 }'); then
  push_url="https://github.com/GoogleCloudPlatform/kubernetes.git"
fi
fetch_remote=$(git remote -v | grep GoogleCloudPlatform/kubernetes.git | grep fetch | awk '{ print $1 }')

VERSION_REGEX="^v(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)$"
[[ ${NEW_VERSION} =~ $VERSION_REGEX ]] || {
  echo "!!! You must specify the version in the form of '$VERSION_REGEX'" >&2
  exit 1
}

VERSION_MAJOR="${BASH_REMATCH[1]}"
VERSION_MINOR="${BASH_REMATCH[2]}"
VERSION_PATCH="${BASH_REMATCH[3]}"

if ! git diff-index --quiet --cached HEAD; then
  echo "!!! You must not have any changes in your index when running this command"
  exit 1
fi

if ! git diff-files --quiet pkg/version/base.go; then
  echo "!!! You have changes in 'pkg/version/base.go' already."
  exit 1
fi

release_branch="release-${VERSION_MAJOR}.${VERSION_MINOR}"
current_branch=$(git rev-parse --abbrev-ref HEAD)

if [[ "${VERSION_PATCH}" != "0" ]]; then
  # sorry, no going back in time, pull latest from upstream
  git remote update > /dev/null 2>&1

  if git ls-remote --tags --exit-code ${fetch_url} refs/tags/${NEW_VERSION} > /dev/null; then
    echo "!!! You are trying to tag ${NEW_VERSION} but it already exists.  Stop it!"
    exit 1
  fi

  last_version="v${VERSION_MAJOR}.${VERSION_MINOR}.$((VERSION_PATCH-1))"
  if ! git ls-remote --tags --exit-code ${fetch_url} refs/tags/${last_version} > /dev/null; then
    echo "!!! You are trying to tag ${NEW_VERSION} but ${last_version} doesn't even exist!"
    exit 1
  fi

  # this is rather magic.  This checks that HEAD is a descendant of the github branch release-x.y
  branches=$(git branch --contains $(git ls-remote --heads ${fetch_url} refs/heads/${release_branch} | cut -f1) ${current_branch})
  if [[ $? -ne 0 ]]; then
    echo "!!! git failed, I dunno...."
    exit 1
  fi

  if [[ ${branches} != "* ${current_branch}" ]]; then
    echo "!!! You are trying to tag to an existing minor release but branch: ${release_branch} is not an ancestor of ${current_branch}"
    exit 1
  fi
fi

SED=sed
if which gsed &>/dev/null; then
  SED=gsed
fi
if ! ("$SED" --version 2>&1 | grep -q GNU); then
  echo "!!! GNU sed is required.  If on OS X, use 'brew install gnu-sed'."
fi

VERSION_FILE="${KUBE_ROOT}/pkg/version/base.go"

GIT_MINOR="${VERSION_MINOR}.${VERSION_PATCH}"
echo "+++ Updating to ${NEW_VERSION}"
"$SED" -r -i -e "s/gitMajor\s+string = \"[^\"]*\"/gitMajor string = \"${VERSION_MAJOR}\"/" "${VERSION_FILE}"
"$SED" -r -i -e "s/gitMinor\s+string = \"[^\"]*\"/gitMinor string = \"${GIT_MINOR}\"/" "${VERSION_FILE}"
"$SED" -r -i -e "s/gitVersion\s+string = \"[^\"]*\"/gitVersion string = \"$NEW_VERSION\"/" "${VERSION_FILE}"
gofmt -s -w "${VERSION_FILE}"

echo "+++ Committing version change"
git add "${VERSION_FILE}"
git commit -m "Kubernetes version $NEW_VERSION"

echo "+++ Tagging version"
git tag -a -m "Kubernetes version $NEW_VERSION" "${NEW_VERSION}"

echo "+++ Updating to ${NEW_VERSION}-dev"
"$SED" -r -i -e "s/gitMajor\s+string = \"[^\"]*\"/gitMajor string = \"${VERSION_MAJOR}\"/" "${VERSION_FILE}"
"$SED" -r -i -e "s/gitMinor\s+string = \"[^\"]*\"/gitMinor string = \"${GIT_MINOR}\+\"/" "${VERSION_FILE}"
"$SED" -r -i -e "s/gitVersion\s+string = \"[^\"]*\"/gitVersion string = \"$NEW_VERSION-dev\"/" "${VERSION_FILE}"
gofmt -s -w "${VERSION_FILE}"

echo "+++ Committing version change"
git add "${VERSION_FILE}"
git commit -m "Kubernetes version ${NEW_VERSION}-dev"

echo ""
echo "Success you must now:"
echo ""
echo "- Push the tag:"
echo "   git push ${push_url} v${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}"
echo "   - Please note you are pushing the tag live BEFORE your PRs."
echo "       You need this so the builds pick up the right tag info."
echo "       If something goes wrong further down please fix the tag!"
echo "       Either delete this tag and give up, fix the tag before your next PR,"
echo "       or find someone who can help solve the tag problem!"
echo ""

if [[ "${VERSION_PATCH}" == "0" ]]; then
  echo "- Send branch: ${current_branch} as a PR to master"
  echo "- Get someone to review and merge that PR"
  echo "- Push the new release branch"
  echo "   git push ${push_url} ${current_branch}:${release_branch}"
else
  echo "- Send branch: ${current_branch} as a PR to ${release_branch}"
  echo "- Get someone to review and merge that PR"
  echo ""
  echo "Now you need to back merge the release branch into master. This should"
  echo "only be done if you are committing to the latest release branch. If the"
  echo "latest release branch is, for example, release-0.10 and you are adding"
  echo "a commit to release-0.9, you may skip the remaining instructions"
  echo ""
  echo "We do this back merge so that master will always show the latest version."
  echo "The version in master would, for exampe show v0.10.2+ instead of v0.10.0+"
  echo "It is not enough to just edit the version file in pkg/version/base.go in a"
  echo "seperate PR. Doing it this way means that git will see the tag you just"
  echo "pushed as an ancestor of master, even though the tag is on on a release"
  echo "branch. The tag will thus be found by tools like git describe"
  echo ""
  echo "- Update so you see that merge in ${fetch_remote}"
  echo "   git remote update"
  echo "- Create and check out a new branch based on master"
  echo "   git checkout -b merge-${release_branch}-to-master ${fetch_remote}/master"
  echo "- Merge the ${release_branch} into your merge-${release_branch}-to-master branch:"
  echo "   git merge -s recursive -X ours ${fetch_remote}/${release_branch}"
  echo "   - It's possible you have merge conflicts that weren't resolved by the merge strategy."
  echo "     - You will almost always want to take what is in HEAD"
  echo "   - If you are not SURE how to solve these correctly, ask for help."
  echo "   - It is possible to break other people's work if you didn't understand why"
  echo "     the conflict happened and the correct way to solve it."
  echo "- Send merge-${release_branch}-to-master as a PR to master"
  echo "- Take the afternoon off"
fi
