#!/bin/bash

# Copyright 2015 The Kubernetes Authors.
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
  : ${JENKINS_GCS_LATEST_PATH:="gs://kubernetes-jenkins/pr-logs/directory"}
  : ${JENKINS_GCS_LOGS_INDIRECT:="gs://kubernetes-jenkins/pr-logs/directory/${JOB_NAME}"}
else
  : ${JENKINS_GCS_LOGS_PATH:="gs://kubernetes-jenkins/logs"}
  : ${JENKINS_GCS_LATEST_PATH:="gs://kubernetes-jenkins/logs"}
  : ${JENKINS_GCS_LOGS_INDIRECT:=""}
fi

readonly artifacts_path="${WORKSPACE}/_artifacts"
readonly gcs_job_path="${JENKINS_GCS_LOGS_PATH}/${JOB_NAME}"
readonly gcs_build_path="${gcs_job_path}/${BUILD_NUMBER}"
readonly gcs_latest_path="${JENKINS_GCS_LATEST_PATH}/${JOB_NAME}"
readonly gcs_indirect_path="${JENKINS_GCS_LOGS_INDIRECT}"
readonly gcs_acl="public-read"
readonly results_url=${gcs_build_path//"gs:/"/"https://console.cloud.google.com/storage/browser"}
readonly timestamp=$(date +%s)

#########################################################################
# $0 is called from different contexts so figure out where kubernetes is.
# Sets non-exported global kubernetes_base_path and defaults to "."
function set_kubernetes_base_path () {
  for kubernetes_base_path in kubernetes go/src/k8s.io/kubernetes .; do
    # Pick a canonical item to find in a kubernetes tree which could be a
    # raw source tree or an expanded tarball.

    [[ -f ${kubernetes_base_path}/cluster/common.sh ]] && break
  done
}

#########################################################################
# Try to discover the kubernetes version.
# prints version
function find_version() {
  (
  # Where are we?
  # This could be set in the global scope at some point if we need to 
  # discover the kubernetes path elsewhere.
  set_kubernetes_base_path

  cd ${kubernetes_base_path}

  if [[ -e "version" ]]; then
    cat version
  elif [[ -e "hack/lib/version.sh" ]]; then
    export KUBE_ROOT="."
    source "hack/lib/version.sh"
    kube::version::get_version_vars
    echo "${KUBE_GIT_VERSION-}"
  else
    # Last resort from the started.json
    gsutil cat ${gcs_build_path}/started.json 2>/dev/null |\
     sed -n 's/ *"version": *"\([^"]*\)",*/\1/p'
  fi
  )
}

function upload_version() {
  local -r version=$(find_version)
  local upload_attempt

  echo -n 'Run starting at '; date -d "@${timestamp}"

  if [[ -n "${version}" ]]; then
    echo "Found Kubernetes version: ${version}"
  else
    echo "Could not find Kubernetes version"
  fi

  local -r json_file="${gcs_build_path}/started.json"
  for upload_attempt in {1..3}; do
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

#########################################################################
# Maintain a single file storing the full build version, Jenkins' job number
# build state.  Limit its size so it does not grow unbounded.
# This is primarily used for and by the
# github.com/kubernetes/release/find_green_build tool.
# @param build_result - the state of the build
#
function update_job_result_cache() {
  local -r build_result=$1
  local -r version=$(find_version)
  local -r job_results=${gcs_job_path}/jobResultsCache.json
  local -r tmp_results="${WORKSPACE}/_tmp/jobResultsCache.tmp"
  local -r cache_size=200
  local upload_attempt

  if [[ -n "${version}" ]]; then
    echo "Found Kubernetes version: ${version}"
  else
    echo "Could not find Kubernetes version"
  fi

  mkdir -p ${tmp_results%/*}

  # Construct a valid json file
  echo "[" > ${tmp_results}

  for upload_attempt in $(seq 3); do
    echo "Copying ${job_results} to ${tmp_results} (attempt ${upload_attempt})"
    # The sed construct below is stripping out only the "version" lines
    # and then ensuring there's a single comma at the end of the line.
    gsutil -q cat ${job_results} 2>&- |\
     sed -n 's/^\({"version".*}\),*/\1,/p' |\
     tail -${cache_size} >> ${tmp_results} || continue
    break
  done

  echo "{\"version\": \"${version}\", \"buildnumber\": \"${BUILD_NUMBER}\"," \
       "\"result\": \"${build_result}\"}" >> ${tmp_results}

  echo "]" >> ${tmp_results}

  for upload_attempt in $(seq 3); do
    echo "Copying ${tmp_results} to ${job_results} (attempt ${upload_attempt})"
    gsutil -q -h "Content-Type:application/json" cp -a "${gcs_acl}" \
           ${tmp_results} ${job_results} || continue
    break
  done

  rm -f ${tmp_results}
}

function upload_artifacts_and_build_result() {
  local -r build_result=$1
  local upload_attempt

  echo -n 'Run finished at '; date -d "@${timestamp}"

  for upload_attempt in {1..3}; do
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

    # For pull jobs, keep a canonical ordering for tools that want to examine
    # the output.
    if [[ "${gcs_indirect_path}" != "" ]]; then
      echo "Writing ${gcs_build_path} to ${gcs_indirect_path}/${BUILD_NUMBER}.txt"
      echo "${gcs_build_path}" | \
        gsutil -q -h "Content-Type:text/plain" \
          cp -a "${gcs_acl}" - "${gcs_indirect_path}/${BUILD_NUMBER}.txt" || continue
      echo "Marking build ${BUILD_NUMBER} as the latest completed build for this PR"
      echo "${BUILD_NUMBER}" | \
        gsutil -q -h "Content-Type:text/plain" -h "Cache-Control:private, max-age=0, no-transform" \
          cp -a "${gcs_acl}" - "${gcs_job_path}/latest-build.txt" || continue
    fi

    # Mark this build as the latest completed.
    echo "Marking build ${BUILD_NUMBER} as the latest completed build"
    echo "${BUILD_NUMBER}" | \
      gsutil -q -h "Content-Type:text/plain" -h "Cache-Control:private, max-age=0, no-transform" \
        cp -a "${gcs_acl}" - "${gcs_latest_path}/latest-build.txt" || continue
    break  # all uploads succeeded if we hit this point
  done

  echo -e "\n\n\n*** View logs and artifacts at ${results_url} ***\n\n"
}

if [[ -n "${JENKINS_BUILD_STARTED:-}" ]]; then
  upload_version
elif [[ -n "${JENKINS_BUILD_FINISHED:-}" ]]; then
  upload_artifacts_and_build_result ${JENKINS_BUILD_FINISHED}
  update_job_result_cache ${JENKINS_BUILD_FINISHED}
else
  echo "Called without JENKINS_BUILD_STARTED or JENKINS_BUILD_FINISHED set."
  echo "Assuming a legacy invocation."
  upload_artifacts_and_build_result "[UNSET]"
fi
