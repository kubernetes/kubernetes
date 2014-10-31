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

# Prerequisites
# TODO (ayurchuk): Perhaps install cloud SDK automagically if we can't find it?

# Exit on any error
set -e

echo "Auto installer for launching Kubernetes"
echo "Release: $RELEASE_PREFIX$RELEASE_NAME"

# Make sure that prerequisites are installed.
if [ "$(which aws)" == "" ]; then
    echo "Can't find aws in PATH, please fix and retry."
    exit 1
fi

# TODO(jbeda): Provide a way to install this in to someplace beyond a temp dir
# so that users have access to local tools.
TMPDIR=$(mktemp -d /tmp/installer.kubernetes.XXXXXX)

cd $TMPDIR

echo "Downloading support files"
aws s3 cp $RELEASE_FULL_PATH/launch-kubernetes.tgz .

tar xzf launch-kubernetes.tgz

./src/scripts/kube-up.sh $RELEASE_FULL_PATH

cd /

# clean up
# rm -rf $TMPDIR
