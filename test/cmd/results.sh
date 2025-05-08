#!/usr/bin/env bash

# Copyright 2022 The Kubernetes Authors.
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

set -o errexit
set -o nounset
set -o pipefail


############################################################
# Kubectl result reporting for different failure scenarios #
############################################################
run_kubectl_results_tests() {
  set -o nounset
  set -o errexit

  kube::log::status "Testing kubectl result output"
  TEMP="${KUBE_TEMP}"
  rm -f "${TEMP}/empty"
  touch "${TEMP}/empty"

  set +o errexit
  kubectl list >"${TEMP}/actual_stdout" 2>"${TEMP}/actual_stderr"
  res=$?
  set -o errexit
  cat >"${TEMP}/expected_stderr" <<EOF
error: unknown command "list" for "kubectl"

Did you mean this?
	get
	wait
EOF
  kube::test::results::diff "${TEMP}/actual_stdout" "${TEMP}/actual_stderr" "$res" "${TEMP}/empty" "${TEMP}/expected_stderr" 1 "kubectl list"

  set +o errexit
  kubectl get pod/no-such-pod >"${TEMP}/actual_stdout" 2>"${TEMP}/actual_stderr"
  res=$?
  set -o errexit
  cat >"${TEMP}/expected_stderr" <<EOF
Error from server (NotFound): pods "no-such-pod" not found
EOF
  kube::test::results::diff "${TEMP}/actual_stdout" "${TEMP}/actual_stderr" "$res" "${TEMP}/empty" "${TEMP}/expected_stderr" 1 "kubectl get pod/no-such-pod"

  output_message=$(kubectl get namespace kube-system 2>&1 "${kube_flags[@]:?}")
  kube::test::if_has_not_string "${output_message}" "command headers turned on"
  output_message=$(kubectl get namespace kube-system 2>&1 "${kube_flags[@]:?}" -v=5)
  kube::test::if_has_string "${output_message}" "command headers turned on"

  set +o nounset
  set +o errexit
}
