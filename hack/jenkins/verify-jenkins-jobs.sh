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

# Pass in Jenkins' URL (localhost:8080 if running this on Jenkins) in $JENKINS.
# Pass in the location of e2e.sh in $E2E.
# This will tell you which tests are in one but not the other.

set -o errexit
set -o nounset
set -o pipefail

# It would be great if each test project went through the same file.
# As it stands, build* go through build.sh, so won't be in e2e.sh.
# check-links goes through verify-linkcheck.sh
# test-go goes through gotest-dockerized.sh
# pull* run on the pull Jenkins instance
readonly EXCEPTIONS='
kubernetes-build
kubernetes-build-1.0
kubernetes-build-1.1
kubernetes-check-links
kubernetes-test-go
kubernetes-pull-build-test-e2e-gce
kubernetes-pull-test-unit-integration
kubernetes-verify-jenkins-jobs
'

# For each element of $1 (needles), searches for it in $2 (haystack). If there
# are missing elements, print them, labelled by $3 (name).
# Does not consider builds in EXCEPTIONS nor non-kubernetes builds.
# Returns exit code 0 on success, 1 on failure.
function search_build() {
  local needles=$1
  local haystack=$2
  local name=$3
  local failed=0
  for build in ${needles}; do
    if ! grep "^kubernetes-" <(echo "${build}") > /dev/null; then
      continue
    fi
    if grep "${build}" <(echo "${EXCEPTIONS}") > /dev/null; then
      continue
    fi
    if ! grep "${build}" <(echo "${haystack}") > /dev/null; then
      if [[ ${failed} -eq 0 ]]; then
        failed=1
        echo "- Builds not found in ${name}:" >&2
      fi
      echo "${build}" >&2
    fi
  done
  return ${failed}
}

# Scrape e2e.sh for build names.
e2e_builds=$(grep "^  kubernetes-.*)$" "${E2E}" | tr -d " )")

# Use Jenkins' JSON API to find test project names.
jenkins_builds=$(curl -sg "${JENKINS}/api/json?tree=jobs[name]&pretty=true" \
  | grep -Po '(?<="name" : ")[^"]*')

search_build "${e2e_builds}" "${jenkins_builds}" "Jenkins" \
  || search_build "${jenkins_builds}" "${e2e_builds}" "e2e.sh"
