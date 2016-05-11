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

# This shell script is called by test-go.sh, when running coverage test
# it accept target paths as input, and it requires the following environment variables:
# KUBE_GO_PACKAGE, KUBE_RACE, KUBE_TIMEOUT, KUBE_COVERMODE,
# GOFLAGS, COVER_REPORT_DIR, COVER_PROFILE, TESTARGS,
# JUNIT_FILENAME_PREFIX, GO_TEST_GREP_PATTERN

PKG=$@
PKG_OUT=${PKG//\//_}
set -o pipefail
set -o nounset
set -o errexit
go test "${GOFLAGS[@]:+${GOFLAGS[@]}}" \
  ${KUBE_RACE} \
  ${KUBE_TIMEOUT} \
  -cover -covermode="${KUBE_COVERMODE}" \
  -coverprofile="${COVER_REPORT_DIR}/${PKG}/${COVER_PROFILE}" \
  "${KUBE_GO_PACKAGE}/${PKG}" \
  "${TESTARGS[@]:+${TESTARGS[@]}}" \
  | tee ${JUNIT_FILENAME_PREFIX:+"${JUNIT_FILENAME_PREFIX}-${PKG_OUT}.stdout"} \
  | grep "${GO_TEST_GREP_PATTERN}"
