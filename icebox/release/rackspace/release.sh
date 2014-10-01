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
KUBE_REPO_ROOT="$(cd "$(dirname "$0")/../../" && pwd -P)"

source "${KUBE_REPO_ROOT}/cluster/kube-env.sh"
source $SCRIPT_DIR/../../cluster/rackspace/${KUBE_CONFIG_FILE-"config-default.sh"}
source $SCRIPT_DIR/../../cluster/rackspace/util.sh

$SCRIPT_DIR/../build-release.sh $INSTANCE_PREFIX

# Copy everything up to swift object store
echo "release/rackspace/release.sh: Uploading to Cloud Files"
if ! swiftly -A $OS_AUTH_URL -U $OS_USERNAME -K $OS_PASSWORD get $CONTAINER > /dev/null 2>&1 ; then
  echo "release/rackspace/release.sh: Container doesn't exist. Creating..."
  swiftly -A $OS_AUTH_URL -U $OS_USERNAME -K $OS_PASSWORD put $CONTAINER > /dev/null 2>&1

fi

for x in master-release.tgz; do
  swiftly -A $OS_AUTH_URL -U $OS_USERNAME -K $OS_PASSWORD put -i _output/release/$x $CONTAINER/output/release/$x > /dev/null 2>&1
done

echo "Release pushed."
