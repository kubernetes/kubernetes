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

# Make bucket and a folder for e2e-node test logs.
# Populate the folder from the logs stored in /tmp/_artifacts/ in the same way as a
# jenkins build would, and then print the URL to view the test results on Gubernator

bucket_name="${USER}-g8r-logs"
echo ""
echo "Using bucket ${bucket_name}"
gsutil mb gs://${bucket_name}/

GCS_JOBS_PATH="gs://${bucket_name}/logs/e2e-node"

TMP_LOG_PATH="/tmp/_artifacts/"
BUILD_LOG_PATH="/tmp/build-log.txt"
readonly gcs_acl="public-read"

if [[ -e "build-log.txt" ]]; then
	echo "Moving build-log.txt"
	mv build-log.txt /tmp/
fi

# Keep build number updated
if ! gsutil -q stat "${GCS_JOBS_PATH}/latest-build.txt"; then
	BUILD_NUMBER=1
else
	BUILD_NUMBER=$(gsutil cat "${GCS_JOBS_PATH}/latest-build.txt")
	let "BUILD_NUMBER += 1"
fi

GCS_LOGS_PATH="gs://${bucket_name}/logs/e2e-node/${BUILD_NUMBER}"

# Upload log files
for f in $TMP_LOG_PATH 
do
	if [[ -d "${TMP_LOG_PATH}" && -n $(ls -A "${TMP_LOG_PATH}") ]]; then
	  echo "Uploading artifacts"
	  gsutil -m -q -o "GSUtil:use_magicfile=True" cp -a "${gcs_acl}" -r -c \
	    -z log,xml,xml "${TMP_LOG_PATH}" "${GCS_LOGS_PATH}/artifacts" || continue
	fi
done

if [[ -e "${BUILD_LOG_PATH}" ]]; then
	echo "Uploading build log"
	gsutil -q cp -Z -a "${gcs_acl}" "${BUILD_LOG_PATH}" "${GCS_LOGS_PATH}"
fi

# Find the k8s version and timestamps needed for started.json and finished.json
version=""
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

start_line=$(grep -n -m 1 "==============" /tmp/build-log.txt |sed 's/\([0-9]*\).*/\1/')
start_time=$(tail -n +${start_line} /tmp/build-log.txt | grep -m 1 -o '[0-9][0-9][0-9][0-9][[:blank:]][0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9]*' | sed 's/^.\{2\}/&\//')
end_time=$(grep -o '[0-9][0-9][0-9][0-9][[:blank:]][0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9]*' /tmp/build-log.txt | tail -1 | sed 's/^.\{2\}/&\//')
start_time_epoch=$(date -d "${start_time}" +%s)
end_time_epoch=$(date -d "${end_time}" +%s)

# Upload started.json and finished.json
echo "Making started.json file"
echo "Run starting at ${start_time}"
json_file="${GCS_LOGS_PATH}/started.json"

for upload_attempt in $(seq 3); do
  echo "Uploading version to: ${json_file} (attempt ${upload_attempt})"
  gsutil -q -h "Content-Type:application/json" cp -a "${gcs_acl}" <(
    echo "{"
    echo "    \"version\": \"${version}\","
    echo "    \"timestamp\": ${start_time_epoch},"
    echo "    \"jenkins-node\": \"${NODE_NAME:-}\""
    echo "}"
  ) "${json_file}" || continue
  break
done

echo "Making finished.json file"
if grep -Fxq "Test Suite Passed" "${BUILD_LOG_PATH}"
	then
		build_result="SUCCESS"
else
		build_result="FAILURE"
fi

echo "Build result is ${build_result}"

for upload_attempt in $(seq 3); do
  echo "Uploading to ${GCS_LOGS_PATH} (attempt ${upload_attempt})"
  gsutil -q -h "Content-Type:application/json" cp -a "${gcs_acl}" <(
    echo "{"
    echo "    \"result\": \"${build_result}\","
    echo "    \"timestamp\": ${end_time_epoch}"
    echo "}"
  ) "${GCS_LOGS_PATH}/finished.json" || continue
  break
 done


# Mark this build as the latest completed.
echo "Marking build ${BUILD_NUMBER} as the latest completed build"
echo "${BUILD_NUMBER}" | \
  gsutil -q -h "Content-Type:text/plain" -h "Cache-Control:private, max-age=0, no-transform" \
    cp -a "${gcs_acl}" - "${GCS_JOBS_PATH}/latest-build.txt"

#TODO: Change localhost:8080 to k8s-gubernator.appspot.com
echo "Gubernator linked below:"
echo "localhost:8080/build/${bucket_name}/logs/e2e-node/${BUILD_NUMBER}?local=on"
