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

# This script will set up the salt directory on the target server.  It takes one
# argument that is a tarball with the pre-compiled kuberntes server binaries.

set -o errexit
set -o nounset
set -o pipefail

readonly SALT_ROOT=$(dirname "${BASH_SOURCE}")

readonly SERVER_BIN_TAR=${1-}
if [[ -z "$SERVER_BIN_TAR" ]]; then
  echo "!!! No binaries specified"
  exit 1
fi

# Create a temp dir for untaring
KUBE_TEMP=$(mktemp -d -t kubernetes.XXXXXX)
trap "rm -rf ${KUBE_TEMP}" EXIT

# This file is meant to run on the master.  It will install the salt configs
# into the appropriate place on the master.

echo "+++ Installing salt files"
mkdir -p /srv
# This bash voodoo will prepend $SALT_ROOT to the start of each item in the
# $SALTDIRS array
readonly SALTDIRS=(salt pillar reactor)
cp -R --preserve=mode "${SALTDIRS[@]/#/${SALT_ROOT}/}" /srv/


echo "+++ Install binaries from tar: $1"
tar -xz -C "${KUBE_TEMP}" -f "$1"
mkdir -p /srv/salt/kube-bins
cp "${KUBE_TEMP}/kubernetes/server/bin/"* /srv/salt/kube-bins/
