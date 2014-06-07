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

# This script will build and release Kubernetes.
#
# The main parameters to this script come from the config.sh file.  This is set
# up by default for development releases.  Feel free to edit it or override some
# of the variables there.

# exit on any error
set -e

source $(dirname $0)/config.sh

cd $(dirname $0)/..

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
cp -r third_party $MASTER_RELEASE_DIR/third_party/go/src

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

echo "Building launch script"
# Create the local install script.  These are the tools to install the local
# tools and launch a new cluster.
LOCAL_RELEASE_DIR=output/release/local-release
mkdir -p $LOCAL_RELEASE_DIR/src/scripts

cp -r cluster/templates $LOCAL_RELEASE_DIR/src/templates
cp -r cluster/*.sh $LOCAL_RELEASE_DIR/src/scripts

tar cz -C $LOCAL_RELEASE_DIR -f output/release/launch-kubernetes.tgz .

echo "#!/bin/bash" >> output/release/launch-kubernetes.sh
echo "RELEASE_TAG=$RELEASE_TAG" >> output/release/launch-kubernetes.sh
echo "RELEASE_PREFIX=$RELEASE_PREFIX" >> output/release/launch-kubernetes.sh
echo "RELEASE_NAME=$RELEASE_NAME" >> output/release/launch-kubernetes.sh
echo "RELEASE_FULL_PATH=$RELEASE_FULL_PATH" >> output/release/launch-kubernetes.sh
cat release/launch-kubernetes-base.sh >> output/release/launch-kubernetes.sh
chmod a+x output/release/launch-kubernetes.sh

# Now copy everything up to the release structure on GS
echo "Uploading to Google Storage"
if ! gsutil ls $RELEASE_BUCKET > /dev/null 2>&1 ; then
  echo "Creating $RELEASE_BUCKET"
  gsutil mb $RELEASE_BUCKET
fi
for x in master-release.tgz launch-kubernetes.tgz launch-kubernetes.sh; do
  gsutil -q cp output/release/$x $RELEASE_FULL_PATH/$x

  make_public_readable $RELEASE_FULL_PATH/$x
done
set_tag $RELEASE_FULL_TAG_PATH $RELEASE_FULL_PATH

echo "Release pushed ($RELEASE_PREFIX$RELEASE_NAME)."

# This isn't quite working right now. Need to figure out packaging the cloudcfg tool.
# echo "  Launch with:"
# echo
# echo "  curl -s -L ${RELEASE_FULL_PATH/gs:\/\//http://storage.googleapis.com/}/launch-kubernetes.sh | bash"
# echo
