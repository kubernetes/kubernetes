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

SCRIPT_DIR=$(CDPATH="" cd $(dirname $0); pwd)

source $SCRIPT_DIR/config.sh
$SCRIPT_DIR/../build-release.sh $INSTANCE_PREFIX

echo "Building launch script"
# Create the local install script.  These are the tools to install the local
# tools and launch a new cluster.
LOCAL_RELEASE_DIR=$SCRIPT_DIR/../../_output/release/local-release
mkdir -p $LOCAL_RELEASE_DIR/src/scripts

cp -r $SCRIPT_DIR/../../cluster/templates $LOCAL_RELEASE_DIR/src/templates
cp -r $SCRIPT_DIR/../../cluster/*.sh $LOCAL_RELEASE_DIR/src/scripts

tar cz -C $LOCAL_RELEASE_DIR -f $SCRIPT_DIR/../../_output/release/launch-kubernetes.tgz .

echo "#!/bin/bash" >> $SCRIPT_DIR/../../_output/release/launch-kubernetes.sh
echo "RELEASE_TAG=$RELEASE_TAG" >> $SCRIPT_DIR/../../_output/release/launch-kubernetes.sh
echo "RELEASE_PREFIX=$RELEASE_PREFIX" >> $SCRIPT_DIR/../../_output/release/launch-kubernetes.sh
echo "RELEASE_NAME=$RELEASE_NAME" >> $SCRIPT_DIR/../../_output/release/launch-kubernetes.sh
echo "RELEASE_FULL_PATH=$RELEASE_FULL_PATH" >> $SCRIPT_DIR/../../_output/release/launch-kubernetes.sh
cat $SCRIPT_DIR/launch-kubernetes-base.sh >> $SCRIPT_DIR/../../_output/release/launch-kubernetes.sh
chmod a+x $SCRIPT_DIR/../../_output/release/launch-kubernetes.sh

echo "Uploading to Amazon S3"
if ! aws s3 ls $RELEASE_BUCKET > /dev/null 2>&1 ; then
    echo "Creating $RELEASE_BUCKET"
    aws s3 mb $RELEASE_BUCKET > /dev/null
fi

aws s3api put-bucket-acl --bucket kubernetes-releases-$AWS_HASH --acl public-read

for x in master-release.tgz launch-kubernetes.tgz launch-kubernetes.sh; do
    aws s3 cp $SCRIPT_DIR/../../_output/release/$x $RELEASE_FULL_PATH/$x > /dev/null
    aws s3api put-object-acl --bucket kubernetes-releases-$AWS_HASH --key $RELEASE_PREFIX$RELEASE_NAME/$x --acl public-read
done

set_tag $RELEASE_FULL_TAG_PATH $RELEASE_FULL_PATH

echo "Release pushed ($RELEASE_PREFIX$RELEASE_NAME)."