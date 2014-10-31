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

# Download and install release

# This script assumes that the environment variable MASTER_RELEASE_TAR contains
# the release tar to download and unpack.  It is meant to be pushed to the
# master and run.

echo "Downloading release ($MASTER_RELEASE_TAR)"
wget $MASTER_RELEASE_TAR


echo "Unpacking release"
rm -rf master-release || false
tar xzf master-release.tgz

echo "Running release install script"
master-release/src/scripts/master-release-install.sh
