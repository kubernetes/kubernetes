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

# This script will build a Kubernetes release tarball

# exit on any error
set -eu
set -o pipefail
IFS=$'\n\t'

SCRIPT_DIR=$(cd $(dirname $0); pwd)

INSTANCE_PREFIX=$1

# First build the release tar.  This gets copied on to the master and installed
# from there.  It includes the go source for the necessary servers along with
# the salt configs.
rm -rf output/release/*

MASTER_RELEASE_DIR=output/release/master-release
mkdir -p $MASTER_RELEASE_DIR/bin
mkdir -p $MASTER_RELEASE_DIR/src/scripts
mkdir -p $MASTER_RELEASE_DIR/third_party/go

echo "Building release tree"
cp release/master-release-install.sh $MASTER_RELEASE_DIR/src/scripts/master-release-install.sh
cp -r cluster/saltbase $MASTER_RELEASE_DIR/src/saltbase

cat << EOF > $MASTER_RELEASE_DIR/src/saltbase/pillar/common.sls
instance_prefix: $INSTANCE_PREFIX-minion
EOF

cp -r third_party/src $MASTER_RELEASE_DIR/third_party/go/src

function find_go_files() {
  find * -not \( \
      \( \
        -wholename 'third_party' \
        -o -wholename 'release' \
      \) -prune \
    \) -name '*.go'
}
for f in $(find_go_files); do
  mkdir -p $MASTER_RELEASE_DIR/src/go/$(dirname ${f})
  cp ${f} ${MASTER_RELEASE_DIR}/src/go/${f}
done

echo "Packaging release"
tar cz -C output/release -f output/release/master-release.tgz master-release
