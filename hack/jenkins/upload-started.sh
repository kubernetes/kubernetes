#!/bin/bash

# Copyright 2016 The Kubernetes Authors.
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


# This is meant to be run at the start of every Jenkins job.
#
# Discovers the local kubernetes version and uploads it, along with
# the current time, to GCS.

set -o errexit
set -o nounset
set -o pipefail

export JENKINS_BUILD_STARTED="true"

echo
echo "Passing through to upload-to-gcs.sh with JENKINS_BUILD_STARTED=${JENKINS_BUILD_STARTED}"
echo "Please update configs to call upload-to-gcs.sh directly."
echo

if [[ -x ./hack/jenkins/upload-to-gcs.sh ]]; then
  ./hack/jenkins/upload-to-gcs.sh
else
  curl -fsS --retry 3 --keepalive-time 2 "https://raw.githubusercontent.com/kubernetes/kubernetes/master/hack/jenkins/upload-to-gcs.sh" | bash -
fi
