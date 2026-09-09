#!/usr/bin/env bash

# Copyright 2018 The Kubernetes Authors.
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

########################################################
# Kubectl version (--client, --output) #
########################################################
run_kubectl_version_tests() {
  set -o nounset
  set -o errexit

  kube::log::status "Testing kubectl version"
  TEMP="${KUBE_TEMP}"

  kubectl get "${kube_flags[@]:?}" --raw /version

  # create version files, one for the client, one for the server.
  # these are the files we will use to ensure that the remainder output is correct
  kube::test::version::object_to_file "Client" "" "${TEMP}/client_version_test"
  kube::test::version::object_to_file "Server" "" "${TEMP}/server_version_test"

  kube::log::status "Testing kubectl version: check client only output matches expected output"
  kube::test::version::object_to_file "Client" "--client" "${TEMP}/client_only_version_test"
  set +e pipefail   # Turn off bash options, since this command pipes empty text.
  kube::test::version::object_to_file "Server" "--client" "${TEMP}/server_client_only_version_test"
  set -e pipefail   # Reset bash options
  kube::test::version::diff_assert "${TEMP}/client_version_test" "eq" "${TEMP}/client_only_version_test" "the flag '--client' shows correct client info"
  kube::test::version::diff_assert "${TEMP}/server_version_test" "ne" "${TEMP}/server_client_only_version_test" "the flag '--client' correctly has no server version info"

  kube::log::status "Testing kubectl version: verify json output"
  kube::test::version::json_client_server_object_to_file "" "clientVersion.gitVersion" "${TEMP}/client_json_version_test"
  kube::test::version::json_client_server_object_to_file "" "serverVersion.gitVersion" "${TEMP}/server_json_version_test"
  kube::test::version::diff_assert "${TEMP}/client_version_test" "eq" "${TEMP}/client_json_version_test" "--output json has correct client info"
  kube::test::version::diff_assert "${TEMP}/server_version_test" "eq" "${TEMP}/server_json_version_test" "--output json has correct server info"

  kube::log::status "Testing kubectl version: verify json output using additional --client flag does not contain serverVersion"
  kube::test::version::json_client_server_object_to_file "--client" "clientVersion.gitVersion" "${TEMP}/client_only_json_version_test"
  kube::test::version::json_client_server_object_to_file "--client" "serverVersion.gitVersion" "${TEMP}/server_client_only_json_version_test"
  kube::test::version::diff_assert "${TEMP}/client_version_test" "eq" "${TEMP}/client_only_json_version_test" "--client --output json has correct client info"
  kube::test::version::diff_assert "${TEMP}/server_version_test" "ne" "${TEMP}/server_client_only_json_version_test" "--client --output json has no server info"

  kube::log::status "Testing kubectl version: compare json output with yaml output"
  kube::test::version::json_object_to_file "" "${TEMP}/client_server_json_version_test"
  kube::test::version::yaml_object_to_file "" "${TEMP}/client_server_yaml_version_test"
  kube::test::version::diff_assert "${TEMP}/client_server_json_version_test" "eq" "${TEMP}/client_server_yaml_version_test" "--output json/yaml has identical information"

  kube::log::status "Testing kubectl version: contains semantic version of embedded kustomize"
  output_message=$(kubectl version)
  kube::test::if_has_not_string "${output_message}" "Kustomize Version\: unknown" "kustomize version should not be unknown"
  kube::test::if_has_string "${output_message}" "Kustomize Version\: v[[:digit:]][[:digit:]]*\.[[:digit:]][[:digit:]]*\.[[:digit:]][[:digit:]]*" "kubectl kustomize version should have a reasonable value"

  kube::log::status "Testing kubectl version: all output formats include kustomize version"
  output_message=$(kubectl version --client)
  kube::test::if_has_string "${output_message}" "Kustomize Version" "kustomize version should be printed when --client is specified"
  output_message=$(kubectl version -o yaml)
  kube::test::if_has_string "${output_message}" "kustomizeVersion" "kustomize version should be printed when -o yaml is used"
  output_message=$(kubectl version -o json)
  kube::test::if_has_string "${output_message}" "kustomizeVersion" "kustomize version should be printed when -o json is used"

  set +o nounset
  set +o errexit
}
