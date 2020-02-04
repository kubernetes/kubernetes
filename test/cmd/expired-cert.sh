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

run_expired_cert_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl valid client cert"

  # Default cluster config does not require TLS, so create a new one
  kubectl config set-cluster test-cluster --server="https://127.0.0.1:${SECURE_API_PORT}"

  local warning_log_string="is expired according to your host's system clock and may be rejected by the server"
  output_message=$(kubectl get --raw /version)
  kube::test::if_has_not_string "${output_message}" "${warning_log_string}"

  kube::log::status "Testing kubectl expired client cert"

  pki_dir=$(mktemp -d)
  openssl genrsa -out "${pki_dir}"/key.pem 512
  openssl req -new -sha256 -key "${pki_dir}"/key.pem -subj "/CN=test" -out "${pki_dir}"/csr.csr
  # generate cert with negative expiration date
  # self-signed because it's invalid anyway
  openssl x509 -req -in "${pki_dir}"/csr.csr -signkey "${pki_dir}"/key.pem -out "${pki_dir}"/cert.pem -days -365 -sha256

  kubectl config set-credentials test --client-key "${pki_dir}"/key.pem --client-certificate "${pki_dir}"/cert.pem
  kubectl config set-context test-context --cluster test-cluster --user test
  old_context=$(kubectl config current-context)
  kubectl config use-context test-context

  # Ignore the return code, we only care about the log output
  output_message=$(kubectl get --raw /version 2>&1 || true)

  kube::test::if_has_string "${output_message}" "${warning_log_string}"

  rm -r "${pki_dir}"
  kubectl config use-context "${old_context}"

  set +o nounset
  set +o errexit
}
