#!/bin/bash

# Copyright 2016 The Kubernetes Authors All rights reserved.
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

# TODO: DRY. Refactor into upload-to-gcs.sh ?
: ${JENKINS_GCS_LOGS_PATH:="gs://kubernetes-jenkins/logs"}
: ${JENKINS_UPLOAD_TO_GCS:="y"}

if [[ ! ${JENKINS_UPLOAD_TO_GCS:-} =~ ^[yY]$ ]]; then
  exit 0
fi

readonly result="$1"
readonly timestamp=$(date +%s)
readonly location="${JENKINS_GCS_LOGS_PATH}/${JOB_NAME}/${BUILD_NUMBER}/finished.json"

echo -n 'Run finished at '; date -d "@${timestamp}"

echo "Uploading build result to: ${location}"
gsutil -q cp -a "public-read" <(
    echo "{"
    echo "    \"result\": \"${result}\","
    echo "    \"timestamp\": ${timestamp}"
    echo "}"
) "${location}"
