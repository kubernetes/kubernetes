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

# Push a new release to the cluster.
#
# This will find the release tar, cause it to be downloaded, unpacked, installed
# and enacted.

# exit on any error
set -e

source $(dirname $0)/util.sh

# Make sure that prerequisites are installed.
for x in gcloud gsutil; do
  if [ "$(which $x)" == "" ]; then
    echo "Can't find $x in PATH, please fix and retry."
    exit 1
  fi
done

# Find the release to use.  Generally it will be passed when doing a 'prod'
# install and will default to the release/config.sh version when doing a
# developer up.
find-release $1

# Detect the project into $PROJECT
detect-master

(
  echo MASTER_RELEASE_TAR=$RELEASE_NORMALIZED/master-release.tgz
  cat $(dirname $0)/templates/download-release.sh
  echo "echo Executing configuration"
  echo "sudo salt '*' mine.update"
  echo "sudo salt --force-color '*' state.highstate"
) | gcloud compute ssh $KUBE_MASTER \
  --project ${PROJECT} --zone ${ZONE} --command="bash"

get-password

echo "Kubernetes cluster is running.  Access the master at:"
echo
echo "  https://${user}:${passwd}@${KUBE_MASTER_IP}"
echo
