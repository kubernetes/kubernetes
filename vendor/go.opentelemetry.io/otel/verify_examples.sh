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

set -euo pipefail

cd $(dirname $0)
TOOLS_DIR=$(pwd)/.tools

if [ -z "${GOPATH}" ] ; then
	printf "GOPATH is not defined.\n"
	exit -1
fi

if [ ! -d "${GOPATH}" ] ; then
	printf "GOPATH ${GOPATH} is invalid \n"
	exit -1
fi

# Pre-requisites
if ! git diff --quiet; then \
	git status
	printf "\n\nError: working tree is not clean\n"
	exit -1
fi

if [ "$(git tag --contains $(git log -1 --pretty=format:"%H"))" = "" ] ; then
	printf "$(git log -1)"
	printf "\n\nError: HEAD is not pointing to a tagged version"
fi

make ${TOOLS_DIR}/gojq

DIR_TMP="${GOPATH}/src/oteltmp/"
rm -rf $DIR_TMP
mkdir -p $DIR_TMP

printf "Copy examples to ${DIR_TMP}\n"
cp -a ./example ${DIR_TMP}

# Update go.mod files
printf "Update go.mod: rename module and remove replace\n"

PACKAGE_DIRS=$(find . -mindepth 2 -type f -name 'go.mod' -exec dirname {} \; | egrep 'example' | sed 's/^\.\///' | sort)

for dir in $PACKAGE_DIRS; do
	printf "  Update go.mod for $dir\n"
	(cd "${DIR_TMP}/${dir}" && \
	 # replaces is ("mod1" "mod2" …)
	 replaces=($(go mod edit -json | ${TOOLS_DIR}/gojq '.Replace[].Old.Path')) && \
	 # strip double quotes
	 replaces=("${replaces[@]%\"}") && \
	 replaces=("${replaces[@]#\"}") && \
	 # make an array (-dropreplace=mod1 -dropreplace=mod2 …)
	 dropreplaces=("${replaces[@]/#/-dropreplace=}") && \
	 go mod edit -module "oteltmp/${dir}" "${dropreplaces[@]}" && \
	 go mod tidy)
done
printf "Update done:\n\n"

# Build directories that contain main package. These directories are different than
# directories that contain go.mod files.
printf "Build examples:\n"
EXAMPLES=$(./get_main_pkgs.sh ./example)
for ex in $EXAMPLES; do
	printf "  Build $ex in ${DIR_TMP}/${ex}\n"
	(cd "${DIR_TMP}/${ex}" && \
	 go build .)
done

# Cleanup
printf "Remove copied files.\n"
rm -rf $DIR_TMP
