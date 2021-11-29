#!/bin/bash

# Copyright The OpenTelemetry Authors
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

help()
{
   printf "\n"
   printf "Usage: $0 -t tag\n"
   printf "\t-t Unreleased tag. Update all go.mod with this tag.\n"
   exit 1 # Exit script after printing help
}

while getopts "t:" opt
do
   case "$opt" in
      t ) TAG="$OPTARG" ;;
      ? ) help ;; # Print help
   esac
done

# Print help in case parameters are empty
if [ -z "$TAG" ]
then
   printf "Tag is missing\n";
   help
fi

# Validate semver
SEMVER_REGEX="^v(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)(\\-[0-9A-Za-z-]+(\\.[0-9A-Za-z-]+)*)?(\\+[0-9A-Za-z-]+(\\.[0-9A-Za-z-]+)*)?$"
if [[ "${TAG}" =~ ${SEMVER_REGEX} ]]; then
	printf "${TAG} is valid semver tag.\n"
else
	printf "${TAG} is not a valid semver tag.\n"
	exit -1
fi

TAG_FOUND=`git tag --list ${TAG}`
if [[ ${TAG_FOUND} = ${TAG} ]] ; then
        printf "Tag ${TAG} already exists\n"
        exit -1
fi

# Get version for version.go
OTEL_VERSION=$(echo "${TAG}" | grep -o '^v[0-9]\+\.[0-9]\+\.[0-9]\+')
# Strip leading v
OTEL_VERSION="${OTEL_VERSION#v}"

cd $(dirname $0)

if ! git diff --quiet; then \
	printf "Working tree is not clean, can't proceed with the release process\n"
	git status
	git diff
	exit 1
fi

# Update version.go
cp ./version.go ./version.go.bak
sed "s/\(return \"\)[0-9]*\.[0-9]*\.[0-9]*\"/\1${OTEL_VERSION}\"/" ./version.go.bak >./version.go
rm -f ./version.go.bak

# Update go.mod
git checkout -b pre_release_${TAG} main
PACKAGE_DIRS=$(find . -mindepth 2 -type f -name 'go.mod' -exec dirname {} \; | egrep -v 'tools' | sed 's/^\.\///' | sort)

for dir in $PACKAGE_DIRS; do
	cp "${dir}/go.mod" "${dir}/go.mod.bak"
	sed "s/opentelemetry.io\/otel\([^ ]*\) v[0-9]*\.[0-9]*\.[0-9]/opentelemetry.io\/otel\1 ${TAG}/" "${dir}/go.mod.bak" >"${dir}/go.mod"
	rm -f "${dir}/go.mod.bak"
done

# Run lint to update go.sum
make lint

# Add changes and commit.
git add .
make ci
git commit -m "Prepare for releasing $TAG"

printf "Now run following to verify the changes.\ngit diff main\n"
printf "\nThen push the changes to upstream\n"
