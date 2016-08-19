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

set -o errexit
set -o nounset
set -o pipefail

source cluster/lib/logging.sh


if [[ $# -eq 0 || ! $1 =~ ^[Yy]$ ]]; then
  read -p "Do you want to run gubernator.sh and upload logs publicly to GCS? [y/n]" yn
  echo
  if [[ ! $yn =~ ^[Yy]$ ]]; then
      exit 1
  fi
fi

# Check that user has gsutil
if [[ $(which gsutil) == "" ]]; then
  echo "Could not find gsutil when running:\which gsutil"
  exit 1
fi

# Check that user has gcloud
if [[ $(which gcloud) == "" ]]; then
  echo "Could not find gcloud when running:\which gcloud"
  exit 1
fi

# Check that user has Credentialed Active account
if ! gcloud auth list | grep -q "ACTIVE"; then
  echo "Could not find active account when running:\gcloud auth list"
  exit 1
fi

readonly gcs_acl="public-read"
bucket_name="${USER}-g8r-logs"
echo ""
V=2 kube::log::status "Using bucket ${bucket_name}"

# Check if the bucket exists
if ! gsutil ls gs:// | grep -q "gs://${bucket_name}/"; then
  V=2 kube::log::status "Creating public bucket ${bucket_name}"
  gsutil mb gs://${bucket_name}/
  # Make all files in the bucket publicly readable
  gsutil acl ch -u AllUsers:R gs://${bucket_name}
else
  V=2 kube::log::status "Bucket already exists"
fi

# Path for e2e-node test results
GCS_JOBS_PATH="gs://${bucket_name}/logs/e2e-node"

ARTIFACTS=${ARTIFACTS:-"/tmp/_artifacts"}
BUILD_LOG_PATH="${ARTIFACTS}/build-log.txt"

if [[ ! -e $BUILD_LOG_PATH ]]; then
  echo "Could not find build-log.txt at ${BUILD_LOG_PATH}"
  exit 1
fi

# Get start and end timestamps based on build-log.txt file contents
# Line where the actual tests start
start_line=$(grep -n -m 1 "^=" ${BUILD_LOG_PATH} | sed 's/\([0-9]*\).*/\1/')
# Create text file starting where the tests start
after_start=$(tail -n +${start_line} ${BUILD_LOG_PATH})
echo "${after_start}" >> build-log-cut.txt
# Match the first timestamp
start_time_raw=$(grep -m 1 -o '[0-9][0-9][0-9][0-9][[:blank:]][0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9]*' build-log-cut.txt)
rm build-log-cut.txt
# Make the date readable by date command (ex: 0101 00:00:00.000 -> 01/01 00:00:00.000)
start_time=$(echo ${start_time_raw} | sed 's/^.\{2\}/&\//')
V=2 kube::log::status "Started at ${start_time}"
# Match the last timestamp in the build-log file
end_time=$(grep -o '[0-9][0-9][0-9][0-9][[:blank:]][0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9]*' ${BUILD_LOG_PATH} | tail -1 | sed 's/^.\{2\}/&\//')
# Convert to epoch time for Gubernator
start_time_epoch=$(date -d "${start_time}" +%s)
end_time_epoch=$(date -d "${end_time}" +%s)

# Make folder name for build from timestamp
BUILD_STAMP=$(echo $start_time | sed 's/\///' | sed 's/ /_/')

GCS_LOGS_PATH="${GCS_JOBS_PATH}/${BUILD_STAMP}"

# Check if folder for same logs already exists
if gsutil ls "${GCS_JOBS_PATH}" | grep -q "${BUILD_STAMP}"; then
  V=2 kube::log::status "Log files already uploaded"
  echo "Gubernator linked below:"
  echo "k8s-gubernator.appspot.com/build/${GCS_LOGS_PATH}?local=on"
  exit
fi

for result in $(find ${ARTIFACTS} -type d -name "results"); do
  if [[ $result != "" && $result != "${ARTIFACTS}/results" && $result != $ARTIFACTS ]]; then
    mv $result/* $ARTIFACTS
  fi
done

# Upload log files
for upload_attempt in $(seq 3); do
  if [[ -d "${ARTIFACTS}" && -n $(ls -A "${ARTIFACTS}") ]]; then
    V=2 kube::log::status "Uploading artifacts"
    gsutil -m -q -o "GSUtil:use_magicfile=True" cp -a "${gcs_acl}" -r -c \
      -z log,xml,xml "${ARTIFACTS}" "${GCS_LOGS_PATH}/artifacts" || continue
  fi
  break
done
for upload_attempt in $(seq 3); do
  if [[ -e "${BUILD_LOG_PATH}" ]]; then
    V=2 kube::log::status "Uploading build log"
    gsutil -q cp -Z -a "${gcs_acl}" "${BUILD_LOG_PATH}" "${GCS_LOGS_PATH}" || continue
  fi
  break
done


# Find the k8s version for started.json
version=""
if [[ -e "version" ]]; then
  version=$(cat "version")
elif [[ -e "hack/lib/version.sh" ]]; then
  export KUBE_ROOT="."
  source "hack/lib/version.sh"
  kube::version::get_version_vars
  version="${KUBE_GIT_VERSION-}"
fi
if [[ -n "${version}" ]]; then
  V=2 kube::log::status "Found Kubernetes version: ${version}"
else
  V=2 kube::log::status "Could not find Kubernetes version"
fi

#Find build result from build-log.txt
if grep -Fxq "Test Suite Passed" "${BUILD_LOG_PATH}"
  then
    build_result="SUCCESS"
else
    build_result="FAILURE"
fi

V=4 kube::log::status "Build result is ${build_result}"

if [[ -e "${ARTIFACTS}/started.json" ]]; then
  rm "${ARTIFACTS}/started.json"
fi

if [[ -e "${ARTIFACTS}/finished.json" ]]; then
  rm "${ARTIFACTS}/finished.json"
fi

V=2 kube::log::status "Constructing started.json and finished.json files"
echo "{" >> "${ARTIFACTS}/started.json"
echo "    \"version\": \"${version}\"," >> "${ARTIFACTS}/started.json"
echo "    \"timestamp\": ${start_time_epoch}," >> "${ARTIFACTS}/started.json"
echo "    \"jenkins-node\": \"${NODE_NAME:-}\"" >> "${ARTIFACTS}/started.json"
echo "}" >> "${ARTIFACTS}/started.json"

echo "{" >> "${ARTIFACTS}/finished.json"
echo "    \"result\": \"${build_result}\"," >> "${ARTIFACTS}/finished.json"
echo "    \"timestamp\": ${end_time_epoch}" >> "${ARTIFACTS}/finished.json"
echo "}" >> "${ARTIFACTS}/finished.json"


# Upload started.json
V=2 kube::log::status "Uploading started.json and finished.json"
V=2 kube::log::status "Run started at ${start_time}"
json_file="${GCS_LOGS_PATH}/started.json"

for upload_attempt in $(seq 3); do
  V=2 kube::log::status "Uploading started.json to ${json_file} (attempt ${upload_attempt})"
  gsutil -q -h "Content-Type:application/json" cp -a "${gcs_acl}" "${ARTIFACTS}/started.json" \
    "${json_file}" || continue
  break
done

# Upload finished.json
for upload_attempt in $(seq 3); do
  V=2 kube::log::status "Uploading finished.json to ${GCS_LOGS_PATH} (attempt ${upload_attempt})"
  gsutil -q -h "Content-Type:application/json" cp -a "${gcs_acl}" "${ARTIFACTS}/finished.json" \
    "${GCS_LOGS_PATH}/finished.json" || continue
  break
done


echo "Gubernator linked below:"
echo "k8s-gubernator.appspot.com/build/${bucket_name}/logs/e2e-node/${BUILD_STAMP}"
