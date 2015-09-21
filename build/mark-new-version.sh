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

VERSION_REGEX="v([0-9]+).([0-9]+(.[0-9]+)?)"
[[ ${NEW_VERSION} =~ $VERSION_REGEX ]] || {
  echo "!!! You must specify the version in the form of '$VERSION_REGEX'" >&2
  exit 1
}

VERSION_MAJOR="${BASH_REMATCH[1]}"
VERSION_MINOR="${BASH_REMATCH[2]}"

if ! git diff-index --quiet --cached HEAD; then
  echo "!!! You must not have any changes in your index when running this command"
  exit 1
fi

if ! git diff-files --quiet pkg/version/base.go; then
  echo "!!! You have changes in 'pkg/version/base.go' already."
  exit 1
fi

SED=sed
if which gsed &>/dev/null; then
  SED=gsed
fi
if ! ("$SED" --version 2>&1 | grep -q GNU); then
  echo "!!! GNU sed is required.  If on OS X, use 'brew install gnu-sed'."
fi

VERSION_FILE="${KUBE_ROOT}/pkg/version/base.go"

echo "+++ Updating to ${NEW_VERSION}"
"$SED" -r -i -e "s/gitMajor\s+string = \"[^\"]*\"/gitMajor string = \"$VERSION_MAJOR\"/" "${VERSION_FILE}"
"$SED" -r -i -e "s/gitMinor\s+string = \"[^\"]*\"/gitMinor string = \"$VERSION_MINOR\"/" "${VERSION_FILE}"
"$SED" -r -i -e "s/gitVersion\s+string = \"[^\"]*\"/gitVersion string = \"$NEW_VERSION\"/" "${VERSION_FILE}"
gofmt -s -w "${VERSION_FILE}"

echo "+++ Committing version change"
git add "${VERSION_FILE}"
git commit -m "Kubernetes version $NEW_VERSION"

echo "+++ Tagging version"
git tag -a -m "Kubernetes version $NEW_VERSION" "${NEW_VERSION}"

echo "+++ Updating to ${NEW_VERSION}-dev"
"$SED" -r -i -e "s/gitMajor\s+string = \"[^\"]*\"/gitMajor string = \"$VERSION_MAJOR\"/" "${VERSION_FILE}"
"$SED" -r -i -e "s/gitMinor\s+string = \"[^\"]*\"/gitMinor string = \"$VERSION_MINOR\+\"/" "${VERSION_FILE}"
"$SED" -r -i -e "s/gitVersion\s+string = \"[^\"]*\"/gitVersion string = \"$NEW_VERSION-dev\"/" "${VERSION_FILE}"
gofmt -s -w "${VERSION_FILE}"

echo "+++ Committing version change"
git add "${VERSION_FILE}"
git commit -m "Kubernetes version ${NEW_VERSION}-dev"
