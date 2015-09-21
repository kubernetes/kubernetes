#!/bin/bash

# Copyright 2014 The Kubernetes Authors All rights reserved.
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

echo "Downloading binary release tar ($SERVER_BINARY_TAR_URL)"
download-or-bust "$SERVER_BINARY_TAR_URL"

echo "Downloading binary release tar ($SALT_TAR_URL)"
download-or-bust "$SALT_TAR_URL"

echo "Unpacking Salt tree"
rm -rf kubernetes
tar xzf "${SALT_TAR_URL##*/}"

echo "Running release install script"
sudo kubernetes/saltbase/install.sh "${SERVER_BINARY_TAR_URL##*/}"
