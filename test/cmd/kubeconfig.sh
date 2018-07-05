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

run_kubectl_config_set_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl(v1:config set)"

  kubectl config set-cluster test-cluster --server="https://does-not-work"

  # Get the api cert and add a comment to avoid flag parsing problems
  cert_data=$(echo "#Comment" && cat "${TMPDIR:-/tmp}/apiserver.crt")

  kubectl config set clusters.test-cluster.certificate-authority-data "$cert_data" --set-raw-bytes
  r_written=$(kubectl config view --raw -o jsonpath='{.clusters[?(@.name == "test-cluster")].cluster.certificate-authority-data}')

  encoded=$(echo -n "$cert_data" | base64)
  kubectl config set clusters.test-cluster.certificate-authority-data "$encoded"
  e_written=$(kubectl config view --raw -o jsonpath='{.clusters[?(@.name == "test-cluster")].cluster.certificate-authority-data}')

  test "$e_written" == "$r_written"

  set +o nounset
  set +o errexit
}

run_client_config_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing client config"

  # Command
  # Pre-condition: kubeconfig "missing" is not a file or directory
  output_message=$(! kubectl get pod --context="" --kubeconfig=missing 2>&1)
  kube::test::if_has_string "${output_message}" "missing: no such file or directory"

  # Pre-condition: kubeconfig "missing" is not a file or directory
  # Command
  output_message=$(! kubectl get pod --user="" --kubeconfig=missing 2>&1)
  # Post-condition: --user contains a valid / empty value, missing config file returns error
  kube::test::if_has_string "${output_message}" "missing: no such file or directory"
  # Command
  output_message=$(! kubectl get pod --cluster="" --kubeconfig=missing 2>&1)
  # Post-condition: --cluster contains a "valid" value, missing config file returns error
  kube::test::if_has_string "${output_message}" "missing: no such file or directory"

  # Pre-condition: context "missing-context" does not exist
  # Command
  output_message=$(! kubectl get pod --context="missing-context" 2>&1)
  kube::test::if_has_string "${output_message}" 'context was not found for specified context: missing-context'
  # Post-condition: invalid or missing context returns error

  # Pre-condition: cluster "missing-cluster" does not exist
  # Command
  output_message=$(! kubectl get pod --cluster="missing-cluster" 2>&1)
  kube::test::if_has_string "${output_message}" 'no server found for cluster "missing-cluster"'
  # Post-condition: invalid or missing cluster returns error

  # Pre-condition: user "missing-user" does not exist
  # Command
  output_message=$(! kubectl get pod --user="missing-user" 2>&1)
  kube::test::if_has_string "${output_message}" 'auth info "missing-user" does not exist'
  # Post-condition: invalid or missing user returns error

  # test invalid config
  kubectl config view | sed -E "s/apiVersion: .*/apiVersion: v-1/g" > "${TMPDIR:-/tmp}"/newconfig.yaml
  output_message=$(! "${KUBE_OUTPUT_HOSTBIN}/kubectl" get pods --context="" --user="" --kubeconfig="${TMPDIR:-/tmp}"/newconfig.yaml 2>&1)
  kube::test::if_has_string "${output_message}" "Error loading config file"

  output_message=$(! kubectl get pod --kubeconfig=missing-config 2>&1)
  kube::test::if_has_string "${output_message}" 'no such file or directory'

  set +o nounset
  set +o errexit
}