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

SCRIPT_DIR=$(CDPATH="" cd $(dirname $0); pwd)

INSTANCE_PREFIX=$1

KUBE_DIR=$SCRIPT_DIR/..

. "${KUBE_DIR}/hack/config-go.sh"

# Next build the release tar.  This gets copied on to the master and installed
# from there.  It includes the go source for the necessary servers along with
# the salt configs.
rm -rf $KUBE_DIR/_output/release/*

MASTER_RELEASE_DIR=$KUBE_DIR/_output/release/master-release
mkdir -p $MASTER_RELEASE_DIR/bin
mkdir -p $MASTER_RELEASE_DIR/src/scripts

echo "Building release tree"
cp $KUBE_DIR/release/master-release-install.sh $MASTER_RELEASE_DIR/src/scripts/master-release-install.sh
cp -r $KUBE_DIR/cluster/saltbase $MASTER_RELEASE_DIR/src/saltbase

# Capture the same version we are using to build the client tools and pass that
# on.
version_ldflags=$(kube::version_ldflags)

# Note: go_opt must stay in sync with the flags in hack/build-go.sh.
cat << EOF > $MASTER_RELEASE_DIR/src/saltbase/pillar/common.sls
instance_prefix: $INSTANCE_PREFIX-minion
go_opt: -ldflags '${version_ldflags}'
EOF

function find_go_files() {
  find * -not \( \
      \( \
        -wholename 'release' \
        -o -wholename 'output' \
        -o -wholename '_output' \
        -o -wholename 'examples' \
        -o -wholename 'test' \
      \) -prune \
    \) -name '*.go'
}
# find_go_files is directory dependent
pushd $KUBE_DIR >/dev/null
for f in $(find_go_files); do
  mkdir -p $MASTER_RELEASE_DIR/src/go/$(dirname ${f})
  cp ${f} ${MASTER_RELEASE_DIR}/src/go/${f}
done
popd >/dev/null

echo "Packaging release"
tar cz -C $KUBE_DIR/_output/release -f $KUBE_DIR/_output/release/master-release.tgz master-release
