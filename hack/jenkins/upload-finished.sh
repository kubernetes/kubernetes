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


# This is meant to be run at the end of every Jenkins job.
#
# Pass in the result of the build in $1. This will upload that, along with the
# current time, to GCS.

set -o errexit
set -o nounset
set -o pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: hack/jenkins/upload-finished.sh RESULT" >&2
    exit 1
fi

export JENKINS_BUILD_FINISHED="$1"

echo
echo "Passing through to upload-to-gcs.sh with JENKINS_BUILD_FINISHED=${JENKINS_BUILD_FINISHED}"
echo "Please update configs to call upload-to-gcs.sh directly."
echo

if [[ -x ./hack/jenkins/upload-to-gcs.sh ]]; then
  ./hack/jenkins/upload-to-gcs.sh
else
  curl -fsS --retry 3 --keepalive-time 2 "https://raw.githubusercontent.com/kubernetes/kubernetes/master/hack/jenkins/upload-to-gcs.sh" | bash -
fi
