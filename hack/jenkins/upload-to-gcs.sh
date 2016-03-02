#!/bin/bash

# Copyright 2015 The Kubernetes Authors All rights reserved.
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

# Run this script in the Jenkins "Execute shell" build action to upload test
# artifacts to GCS. Since it uses gsutil directly, it's a bit faster at
# uploading large numbers of files than the GCS Jenkins plugin.
# We also intentionally ignore gsutil errors since we don't want failed uploads
# to fail the entire test run (#13548).

# Note: for magicfile support to work correctly, the "file" utility must be
# installed.

# TODO: eventually fold this all into upload-finished.sh, once every job is
# using it.

set -o errexit
set -o nounset
set -o pipefail

if [[ ${JOB_NAME} =~ -pull- ]]; then
  : ${JENKINS_GCS_LOGS_PATH:="gs://kubernetes-jenkins/pr-logs/pull/${ghprbPullId:-unknown}"}
else
  : ${JENKINS_GCS_LOGS_PATH:="gs://kubernetes-jenkins/logs"}
fi

readonly artifacts_path="${WORKSPACE}/_artifacts"
readonly gcs_job_path="${JENKINS_GCS_LOGS_PATH}/${JOB_NAME}"
readonly gcs_build_path="${gcs_job_path}/${BUILD_NUMBER}"
readonly gcs_acl="public-read"

for upload_attempt in $(seq 3); do
  echo "Uploading to ${gcs_build_path} (attempt ${upload_attempt})"
  if [[ -d "${artifacts_path}" && -n $(ls -A "${artifacts_path}") ]]; then
    gsutil -m -q -o "GSUtil:use_magicfile=True" cp -a "${gcs_acl}" -r -c \
      -z log,txt,xml "${artifacts_path}" "${gcs_build_path}/artifacts" || continue
  fi
  # Mark this build as the latest completed.
  echo "${BUILD_NUMBER}" | \
    gsutil -q -h "Content-Type:text/plain" -h "Cache-Control:private, max-age=0, no-transform" \
      cp -a "${gcs_acl}" - "${gcs_job_path}/latest-build.txt" || continue
  break  # all uploads succeeded if we hit this point
done

readonly results_url=${gcs_build_path//"gs:/"/"https://storage.cloud.google.com"}
echo -e "\n\n\n*** View logs and artifacts at ${results_url} ***\n\n"
