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
KUBE_REPO_ROOT="$(cd "$(dirname "$0")/../" && pwd -P)"

source "${KUBE_REPO_ROOT}/cluster/kube-env.sh"
source $(dirname ${BASH_SOURCE})/../cluster/${KUBERNETES_PROVIDER}/${KUBE_CONFIG_FILE-"config-default.sh"}

cd $SCRIPT_DIR/..

$SCRIPT_DIR/build-release.sh $INSTANCE_PREFIX

echo "Building launch script"
# Create the local install script.  These are the tools to install the local
# tools and launch a new cluster.
LOCAL_RELEASE_DIR=_output/release/local-release
mkdir -p $LOCAL_RELEASE_DIR/src/scripts

cp -r cluster/templates $LOCAL_RELEASE_DIR/src/templates
cp -r cluster/*.sh $LOCAL_RELEASE_DIR/src/scripts

tar cz -C $LOCAL_RELEASE_DIR -f _output/release/launch-kubernetes.tgz .

echo "#!/bin/bash" >> _output/release/launch-kubernetes.sh
echo "RELEASE_TAG=$RELEASE_TAG" >> _output/release/launch-kubernetes.sh
echo "RELEASE_PREFIX=$RELEASE_PREFIX" >> _output/release/launch-kubernetes.sh
echo "RELEASE_NAME=$RELEASE_NAME" >> _output/release/launch-kubernetes.sh
echo "RELEASE_FULL_PATH=$RELEASE_FULL_PATH" >> _output/release/launch-kubernetes.sh
cat release/launch-kubernetes-base.sh >> _output/release/launch-kubernetes.sh
chmod a+x _output/release/launch-kubernetes.sh

# Now copy everything up to the release structure on GS
echo "Uploading to Google Storage"
if ! gsutil ls $RELEASE_BUCKET > /dev/null 2>&1 ; then
  echo "Creating $RELEASE_BUCKET"
  gsutil mb $RELEASE_BUCKET
fi
for x in master-release.tgz launch-kubernetes.tgz launch-kubernetes.sh; do
  gsutil -q cp _output/release/$x $RELEASE_FULL_PATH/$x

  make_public_readable $RELEASE_FULL_PATH/$x
done
set_tag $RELEASE_FULL_TAG_PATH $RELEASE_FULL_PATH

echo "Release pushed ($RELEASE_PREFIX$RELEASE_NAME)."

# This isn't quite working right now. Need to figure out packaging the kubecfg tool.
# echo "  Launch with:"
# echo
# echo "  curl -s -L ${RELEASE_FULL_PATH/gs:\/\//http://storage.googleapis.com/}/launch-kubernetes.sh | bash"
# echo
