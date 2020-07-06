#!/usr/bin/env bash

# Copyright 2020 The Kubernetes Authors.
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

run_exec_credentials_tests() {
  set -o nounset
  set -o errexit

  kube::log::status "Testing kubectl with configured exec credentials plugin"

  cat > "${TMPDIR:-/tmp}"/invalid_exec_plugin.yaml << EOF
apiVersion: v1
clusters:
- cluster:
  name: test
contexts:
- context:
    cluster: test
    user: invalid_token_user
  name: test
current-context: test
kind: Config
preferences: {}
users:
- name: invalid_token_user
  user:
    exec:
      apiVersion: client.authentication.k8s.io/v1beta1
      # Any invalid exec credential plugin will do to demonstrate
      command: ls
EOF

  ### Provided --token should take precedence, thus not triggering the (invalid) exec credential plugin
  # Pre-condition: Client certificate authentication enabled on the API server
  kube::util::test_client_certificate_authentication_enabled
  # Command
  output=$(kubectl "${kube_flags_with_token[@]:?}" --kubeconfig="${TMPDIR:-/tmp}"/invalid_exec_plugin.yaml get namespace kube-system -o name || true)

  if [[ "${output}" == "namespace/kube-system" ]]; then
    kube::log::status "exec credential plugin not triggered since kubectl was called with provided --token"
  else
    kube::log::status "Unexpected output when providing --token for authentication - exec credential plugin likely triggered. Output: ${output}"
    exit 1    
  fi
  # Post-condition: None

  ### Without provided --token, the exec credential plugin should be triggered
  # Pre-condition: Client certificate authentication enabled on the API server - already checked by positive test above

  # Command
  output2=$(kubectl "${kube_flags_without_token[@]:?}" --kubeconfig="${TMPDIR:-/tmp}"/invalid_exec_plugin.yaml get namespace kube-system -o name 2>&1 || true)

  if [[ "${output2}" =~ "json parse error" ]]; then
    kube::log::status "exec credential plugin triggered since kubectl was called without provided --token"
  else
    kube::log::status "Unexpected output when not providing --token for authentication - exec credential plugin not triggered. Output: ${output2}"
    exit 1
  fi
  # Post-condition: None

  rm "${TMPDIR:-/tmp}"/invalid_exec_plugin.yaml

  set +o nounset
  set +o errexit
}
