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
declare -r TMPDIR=${TMPDIR:-"/tmp"}
declare -r KUBE_RELEASE_UMASK=${KUBE_RELEASE_UMASK:-022}

VERSION_REGEX="^v(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)(-(beta|alpha)\\.(0|[1-9][0-9]*))?$"
[[ ${KUBE_RELEASE_VERSION} =~ ${VERSION_REGEX} ]] || {
  echo "!!! You must specify the version you are releasing in the form of '${VERSION_REGEX}'" >&2
  exit 1
}
VERSION_MAJOR="${BASH_REMATCH[1]}"
VERSION_MINOR="${BASH_REMATCH[2]}"
if [[ "$KUBE_RELEASE_VERSION" =~ "alpha" ]]; then
  # We don't want to version docs for alpha releases, so we are just pointing to head.
  RELEASE_BRANCH="master"
else
  RELEASE_BRANCH="release-${VERSION_MAJOR}.${VERSION_MINOR}"
fi

if [[ "$KUBE_RELEASE_VERSION" =~ alpha|beta ]]; then
  KUBE_RELEASE_TYPE="latest"
else
  KUBE_RELEASE_TYPE="stable"
fi

declare -r KUBE_BUILD_DIR=$(mktemp -d "${TMPDIR}/kubernetes-build-release-${KUBE_RELEASE_VERSION}-XXXXXXX")

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
# In order to build docker images for a release and tag them appropriately we need
# to set these two variables.
export KUBE_DOCKER_REGISTRY="gcr.io/google_containers"
export KUBE_DOCKER_IMAGE_TAG="${KUBE_RELEASE_VERSION}"

make release

if ${KUBE_BUILD_DIR}/cluster/kubectl.sh version | grep Client | grep dirty; then
  echo "!!! Tag at invalid point, or something else is bad. Build is dirty. Don't push this build." >&2
  exit 1
fi

ln -s ${KUBE_BUILD_DIR}/_output/release-tars/kubernetes.tar.gz ${KUBE_BUILD_DIR}

MD5=$(md5 "${KUBE_BUILD_DIR}/kubernetes.tar.gz")
SHA1=$(sha1 "${KUBE_BUILD_DIR}/kubernetes.tar.gz")

cat <<- EOM

Success!  You must now do the following (you may want to cut and paste these
instructions elsewhere):

  1) pushd ${KUBE_BUILD_DIR}; build/push-official-release.sh ${KUBE_RELEASE_VERSION} ${KUBE_RELEASE_TYPE}

  2) Release notes draft, to be published when the release is announced:

     a) Title:

       Release ${KUBE_RELEASE_VERSION}

     b) Template for the description:

## [Documentation](http://releases.k8s.io/${RELEASE_BRANCH}/docs/README.md)
## [Examples](http://releases.k8s.io/${RELEASE_BRANCH}/examples)
## Changes since <last release> (last PR <last PR>)

<release notes>

binary | hash alg | hash
------ | -------- | ----
\`kubernetes.tar.gz\` | md5 | \`${MD5}\`
\`kubernetes.tar.gz\` | sha1 | \`${SHA1}\`

  3) Ensure all the binaries are in place on GCS before cleaning, (you might
  want to wait until the release is announced and published on GitHub, too).

  4) make clean; popd; rm -rf ${KUBE_BUILD_DIR}

EOM
