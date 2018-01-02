#!/usr/bin/env bash

# Copyright 2015 Google Inc. All rights reserved.
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

set -e

VERSION=$( git describe --tags --dirty --abbrev=14 | sed -E 's/-([0-9]+)-g/.\1+/' )
# Only allow releases of tagged versions.
TAGGED='^v[0-9]+\.[0-9]+\.[0-9]+(-(alpha|beta)[0-9]*)?$'
if [[ ! "$VERSION" =~ $TAGGED ]]; then
  echo "Error: Only tagged versions are allowed for releases" >&2
  echo "Found: $VERSION" >&2
  exit 1
fi

# Don't include hostname with release builds
if ! git_user="$(git config --get user.email)"; then
  echo "Error: git user not set, use:"
  echo "git config user.email <email>"
  exit 1
fi

# Build the release.
export BUILD_USER="$git_user"
export BUILD_DATE=$( date +%Y%m%d ) # Release date is only to day-granularity
export GO_CMD="build" # Don't use cached build objects for releases.
export VERBOSE=true
build/build.sh

# Build the docker image
echo ">> building cadvisor docker image"
docker_tag="google/cadvisor:$VERSION"
gcr_tag="gcr.io/google_containers/cadvisor:$VERSION"
docker build -t $docker_tag -t $gcr_tag -f deploy/Dockerfile .

echo
echo "Release info:"
echo "VERSION=$VERSION"
sha256sum --tag cadvisor
echo "docker image: $docker_tag"
echo "gcr.io image: $gcr_tag"

exit 0
