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

# Bumps the version number by creating a couple of commits.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..

NEW_VERSION=${1-}

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
  if [[ ${current_branch} != "${release_branch}" ]]; then
    echo "!!! You are trying to tag to an existing minor release but are not on the release branch: ${release_branch}"
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

if [[ "${VERSION_PATCH}" == "0" ]]; then
  echo "+++ Creating release branch"
  git branch "${release_branch}"
fi

echo "Success you must now:"
echo ""
echo "- Push the tag:"
echo "   git push git@github.com:GoogleCloudPlatform/kubernetes.git v${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}"
if [[ "${VERSION_PATCH}" == "0" ]]; then
  echo "- Submit branch: ${current_branch} as a PR to master"
  echo "- Merge that PR"
  echo "- Push the new release branch"
  echo "   git push git@github.com:GoogleCloudPlatform/kubernetes.git ${release_branch}"
else
  echo "- Submit branch: ${current_branch} as a PR to ${release_branch}"
  echo "- Merge that PR"
fi
