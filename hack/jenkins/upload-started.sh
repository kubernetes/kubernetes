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


# This is meant to be run at the start of every Jenkins job.
#
# Discovers the local kubernetes version and uploads it, along with
# the current time, to GCS.

set -o errexit
set -o nounset
set -o pipefail

# TODO: DRY. Refactor into upload-to-gcs.sh ?
: ${JENKINS_GCS_LOGS_PATH:="gs://kubernetes-jenkins/logs"}
: ${JENKINS_UPLOAD_TO_GCS:="y"}

if [[ ! ${JENKINS_UPLOAD_TO_GCS:-} =~ ^[yY]$ ]]; then
  exit 0
fi

version=""
readonly timestamp=$(date +%s)
readonly location="${JENKINS_GCS_LOGS_PATH}/${JOB_NAME}/${BUILD_NUMBER}/started.json"

echo -n 'Run starting at '; date -d "@${timestamp}"

# Try to discover the kubernetes version.
if [[ -e "version" ]]; then
    version=$(cat "version")
elif [[ -e "hack/lib/version.sh" ]]; then
    version=$(
        export KUBE_ROOT="."
        source "hack/lib/version.sh"
        kube::version::get_version_vars
        echo "${KUBE_GIT_VERSION-}"
    )
fi

if [[ -n "${version}" ]]; then
    echo "Found Kubernetes version: ${version}"
else
    echo "Could not find Kubernetes version"
fi

echo "Uploading version to: ${location}"
gsutil -q cp -a "public-read" <(
    echo "{"
    echo "    \"version\": \"${version}\","
    echo "    \"timestamp\": ${timestamp}"
    echo "}"
) "${location}"
