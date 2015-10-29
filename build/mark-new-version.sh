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

if [ "$#" -ne 1 ]; then
  echo "Usage: ${0} <version>"
  exit 1
fi

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..

NEW_VERSION=${1-}

fetch_url=$(git remote -v | grep kubernetes/kubernetes.git | grep fetch | awk '{ print $2 }')
if ! push_url=$(git remote -v | grep kubernetes/kubernetes.git | grep push | awk '{ print $2 }'); then
  push_url="https://github.com/kubernetes/kubernetes.git"
fi
fetch_remote=$(git remote -v | grep kubernetes/kubernetes.git | grep fetch | awk '{ print $1 }')

VERSION_REGEX="^v(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)$"
[[ ${NEW_VERSION} =~ $VERSION_REGEX ]] || {
  echo "!!! You must specify the version in the form of '$VERSION_REGEX'" >&2
  exit 1
}

VERSION_MAJOR="${BASH_REMATCH[1]}"
VERSION_MINOR="${BASH_REMATCH[2]}"
VERSION_PATCH="${BASH_REMATCH[3]}"

if ! git diff HEAD --quiet; then
  echo "!!! You must not have any uncommitted changes when running this command"
  exit 1
fi

if ! git diff-files --quiet pkg/version/base.go; then
  echo "!!! You have changes in 'pkg/version/base.go' already."
  exit 1
fi

release_branch="release-${VERSION_MAJOR}.${VERSION_MINOR}"
current_branch=$(git rev-parse --abbrev-ref HEAD)
head_commit=$(git rev-parse --short HEAD)

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
if ! ($SED --version 2>&1 | grep -q GNU); then
  echo "!!! GNU sed is required.  If on OS X, use 'brew install gnu-sed'."
fi

echo "+++ Running ./versionize-docs"
# Links in docs should always point to the release branch. 
${KUBE_ROOT}/build/versionize-docs.sh ${release_branch}

echo "+++ Updating swagger"
${KUBE_ROOT}/hack/update-generated-swagger-docs.sh

git commit -am "Versioning docs and examples to ${release_branch}"

VERSION_FILE="${KUBE_ROOT}/pkg/version/base.go"

GIT_MINOR="${VERSION_MINOR}.${VERSION_PATCH}"
echo "+++ Updating to ${NEW_VERSION}"
$SED -ri -e "s/gitMajor\s+string = \"[^\"]*\"/gitMajor string = \"${VERSION_MAJOR}\"/" "${VERSION_FILE}"
$SED -ri -e "s/gitMinor\s+string = \"[^\"]*\"/gitMinor string = \"${GIT_MINOR}\"/" "${VERSION_FILE}"
$SED -ri -e "s/gitVersion\s+string = \"[^\"]*\"/gitVersion string = \"$NEW_VERSION-${release_branch}+\$Format:%h\$\"/" "${VERSION_FILE}"
gofmt -s -w "${VERSION_FILE}"


echo "+++ Committing version change"
git add "${VERSION_FILE}"
git commit -m "Kubernetes version ${NEW_VERSION}"

echo "+++ Tagging version"
git tag -a -m "Kubernetes version ${NEW_VERSION}" "${NEW_VERSION}"
# We have to sleep for a bit so that the timestamp of the beta tag is after the
# timestamp of the release version, so that future commits are described as
# beta, and not release versions.
echo "+++ Waiting for 5 seconds to ensure timestamps are different before continuing"
sleep 5
echo "+++ Tagging beta tag"
declare -r beta_ver="v${VERSION_MAJOR}.${VERSION_MINOR}.$((${VERSION_PATCH}+1))-beta"
git tag -a -m "Kubernetes version ${beta_ver}" "${beta_ver}"
newtag=$(git rev-parse --short HEAD)

if [[ "${VERSION_PATCH}" == "0" ]]; then
  declare -r alpha_ver="v${VERSION_MAJOR}.$((${VERSION_MINOR}+1)).0-alpha.0"
  git tag -a -m "Kubernetes pre-release branch ${alpha_ver}" "${alpha_ver}" "${head_commit}"
fi

echo ""
echo "Success you must now:"
echo ""
echo "- Push the tags:"
echo "   git push ${push_url} ${NEW_VERSION}"
echo "   git push ${push_url} ${beta_ver}"

if [[ "${VERSION_PATCH}" == "0" ]]; then
  echo "- Push the alpha tag:"
  echo "   git push ${push_url} ${alpha_ver}"
  echo "- Push the new release branch:"
  echo "   git push ${push_url} ${current_branch}:${release_branch}"
  echo "- DO NOTHING TO MASTER. You were done with master when you pushed the alpha tag."
else
  echo "- Send branch: ${current_branch} as a PR to ${release_branch} <-- NOTE THIS"
  echo "- In the contents of the PR, include the PRs in the release:"
  echo "    hack/cherry_pick_list.sh ${current_branch}^1"
  echo "  This helps cross-link PRs to patch releases they're part of in GitHub."
  echo "- Have someone review the PR. This is a mechanical review to ensure it contains"
  echo "  the ${NEW_VERSION} commit, which was tagged at ${newtag}."
fi
