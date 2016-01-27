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

# Builds an official release based on a git tag, with instructions for
# how to proceed after the bits are built.

set -o errexit
set -o nounset
set -o pipefail

# Get the md5 (duplicated from common.sh, but don't want to pull in
# all of common.sh here)
function md5() {
  if which md5 >/dev/null 2>&1; then
    md5 -q "$1"
  else
    md5sum "$1" | awk '{ print $1 }'
  fi
}

# Get the sha1 (duplicated from common.sh, but don't want to pull in
# all of common.sh here)
function sha1() {
  if which shasum >/dev/null 2>&1; then
    shasum -a1 "$1" | awk '{ print $1 }'
  else
    sha1sum "$1" | awk '{ print $1 }'
  fi
}

declare -r KUBE_GITHUB="https://github.com/kubernetes/kubernetes.git"
declare -r KUBE_RELEASE_VERSION=${1-}
declare -r KUBE_RELEASE_UMASK=${KUBE_RELEASE_UMASK:-022}

VERSION_REGEX="^v(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)$"
[[ ${KUBE_RELEASE_VERSION} =~ ${VERSION_REGEX} ]] || {
  echo "!!! You must specify the version you are releasing in the form of '${VERSION_REGEX}'" >&2
  exit 1
}

declare -r KUBE_BUILD_DIR="/tmp/kubernetes-release-${KUBE_RELEASE_VERSION}-$(date +%s)"

# Set the default umask for the release. This ensures consistency
# across our release builds.
umask "${KUBE_RELEASE_UMASK}"

echo "Cloning ${KUBE_GITHUB} at ${KUBE_RELEASE_VERSION}."
echo
echo "NOTE: Ignore the deatched HEAD warning you're about to get. We want that."
echo
git clone ${KUBE_GITHUB} -b "${KUBE_RELEASE_VERSION}" "${KUBE_BUILD_DIR}"

# !!! REMINDER !!!
#
# Past this point, you are dealing with a different release. Don't
# assume you're executing code from the same repo as this script is
# running in. This needs to be a version agnostic build.

echo
echo "Cloned, building release."
echo

cd "${KUBE_BUILD_DIR}"
export KUBE_RELEASE_RUN_TESTS=n
export KUBE_SKIP_CONFIRMATIONS=y
make release

if ${KUBE_BUILD_DIR}/cluster/kubectl.sh version | grep Client | grep dirty; then
  echo "!!! Tag at invalid point, or something else is bad. Build is dirty. Don't push this build." >&2
  exit 1
fi

ln -s ${KUBE_BUILD_DIR}/_output/release-tars/kubernetes.tar.gz ${KUBE_BUILD_DIR}

MD5=$(md5 "${KUBE_BUILD_DIR}/kubernetes.tar.gz")
SHA1=$(sha1 "${KUBE_BUILD_DIR}/kubernetes.tar.gz")

echo ""
echo "Success! You must now do the following: (you may want to cut"
echo "  and paste these instructions elsewhere, step 1 can be spammy)"
echo ""
echo "  1) (cd ${KUBE_BUILD_DIR}; build/push-official-release.sh ${KUBE_RELEASE_VERSION})"
echo "  2) Go to https://github.com/GoogleCloudPlatform/kubernetes/releases"
echo "     and create a new 'Release ${KUBE_RELEASE_VERSION} Candidate' release"
echo "     with the ${KUBE_RELEASE_VERSION} tag. Mark it as a pre-release."
echo "  3) Upload the ${KUBE_BUILD_DIR}/kubernetes.tar.gz to GitHub"
echo "  4) Use this template for the release:"
echo ""
echo "## [Documentation](http://releases.k8s.io/${KUBE_RELEASE_VERSION}/docs/README.md)"
echo "## [Examples](http://releases.k8s.io/${KUBE_RELEASE_VERSION}/examples)"
echo "## Changes since <last release> (last PR <last PR>)"
echo ""
echo "<release notes>"
echo ""
echo "binary | hash alg | hash"
echo "------ | -------- | ----"
echo "\`kubernetes.tar.gz\` | md5 | \`${MD5}\`"
echo "\`kubernetes.tar.gz\` | sha1 | \`${SHA1}\`"
echo ""
echo "     We'll fill in the release notes in the next stage."
echo "  5) Ensure all the binaries are in place on GitHub and GCS before cleaning."
echo "  6) (cd ${KUBE_BUILD_DIR}; make clean; cd -; rm -rf ${KUBE_BUILD_DIR})"
echo ""
