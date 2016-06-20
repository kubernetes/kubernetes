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

# This script uploads metadata and test results to Google Cloud Storage, in the
# location indicated by JENKINS_GCS_LOGS_PATH. By default, we use the Google
# kubernetes-jenkins bucket.
#
# The script looks for one of two environment variables to be set:
#   JENKINS_BUILD_STARTED: set to a nonempty string to upload version
#     information to 'started.json'. The value of the variable is not
#     currently used.
#   JENKINS_BUILD_FINISHED: set to the Jenkins build result to upload the build
#     result to 'finished.json', any test artifacts, and update the
#     'latest-build.txt' file pointer. Since this script uses gsutil directly,
#     it's a bit faster at uploading large numbers of files than the GCS Jenkins
#     plugin. It also makes use of gsutil's gzip functionality.
#
# Note: for magicfile support to work correctly, the "file" utility must be
# installed.

set -o errexit
set -o nounset
set -o pipefail

if [[ -n "${JENKINS_BUILD_STARTED:-}" && -n "${JENKINS_BUILD_FINISHED:-}" ]]; then
  echo "Error: JENKINS_BUILD_STARTED and JENKINS_BUILD_FINISHED should not both be set!"
  exit 1
fi

if [[ ! ${JENKINS_UPLOAD_TO_GCS:-y} =~ ^[yY]$ ]]; then
  exit 0
fi

if [[ ${JOB_NAME} =~ -pull- ]]; then
  : ${JENKINS_GCS_LOGS_PATH:="gs://kubernetes-jenkins/pr-logs/pull/${ghprbPullId:-unknown}"}
else
  : ${JENKINS_GCS_LOGS_PATH:="gs://kubernetes-jenkins/logs"}
fi

readonly artifacts_path="${WORKSPACE}/_artifacts"
readonly gcs_job_path="${JENKINS_GCS_LOGS_PATH}/${JOB_NAME}"
readonly gcs_build_path="${gcs_job_path}/${BUILD_NUMBER}"
readonly gcs_acl="public-read"
readonly results_url=${gcs_build_path//"gs:/"/"https://console.cloud.google.com/storage/browser"}
readonly timestamp=$(date +%s)

function upload_version() {
  echo -n 'Run starting at '; date -d "@${timestamp}"

  # Try to discover the kubernetes version.
  local version=""
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

  local -r json_file="${gcs_build_path}/started.json"
  for upload_attempt in $(seq 3); do
    echo "Uploading version to: ${json_file} (attempt ${upload_attempt})"
    gsutil -q -h "Content-Type:application/json" cp -a "${gcs_acl}" <(
      echo "{"
      echo "    \"version\": \"${version}\","
      echo "    \"timestamp\": ${timestamp},"
      echo "    \"jenkins-node\": \"${NODE_NAME:-}\""
      echo "}"
    ) "${json_file}" || continue
    break
  done
}

function upload_artifacts_and_build_result() {
  local -r build_result=$1
  echo -n 'Run finished at '; date -d "@${timestamp}"

  for upload_attempt in $(seq 3); do
    echo "Uploading to ${gcs_build_path} (attempt ${upload_attempt})"
    echo "Uploading build result: ${build_result}"
    gsutil -q -h "Content-Type:application/json" cp -a "${gcs_acl}" <(
      echo "{"
      echo "    \"result\": \"${build_result}\","
      echo "    \"timestamp\": ${timestamp}"
      echo "}"
    ) "${gcs_build_path}/finished.json" || continue
    if [[ -d "${artifacts_path}" && -n $(ls -A "${artifacts_path}") ]]; then
      echo "Uploading artifacts"
      gsutil -m -q -o "GSUtil:use_magicfile=True" cp -a "${gcs_acl}" -r -c \
        -z log,txt,xml "${artifacts_path}" "${gcs_build_path}/artifacts" || continue
    fi
    if [[ -e "${WORKSPACE}/build-log.txt" ]]; then
      echo "Uploading build log"
      gsutil -q cp -Z -a "${gcs_acl}" "${WORKSPACE}/build-log.txt" "${gcs_build_path}"
    fi
    # Mark this build as the latest completed.
    echo "Marking build ${BUILD_NUMBER} as the latest completed build"
    echo "${BUILD_NUMBER}" | \
      gsutil -q -h "Content-Type:text/plain" -h "Cache-Control:private, max-age=0, no-transform" \
        cp -a "${gcs_acl}" - "${gcs_job_path}/latest-build.txt" || continue
    break  # all uploads succeeded if we hit this point
  done

  echo -e "\n\n\n*** View logs and artifacts at ${results_url} ***\n\n"
}

if [[ -n "${JENKINS_BUILD_STARTED:-}" ]]; then
  upload_version
elif [[ -n "${JENKINS_BUILD_FINISHED:-}" ]]; then
  upload_artifacts_and_build_result ${JENKINS_BUILD_FINISHED}
else
  echo "Called without JENKINS_BUILD_STARTED or JENKINS_BUILD_FINISHED set."
  echo "Assuming a legacy invocation."
  upload_artifacts_and_build_result "[UNSET]"
fi
