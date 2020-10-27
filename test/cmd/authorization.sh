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

run_authorization_tests() {
  set -o nounset
  set -o errexit

  kube::log::status "Testing authorization"

  # check remote authorization endpoint, kubectl doesn't actually display the returned object so this isn't super useful
  # but it proves that works
  kubectl create -f test/fixtures/pkg/kubectl/cmd/create/sar-v1.json --validate=false
  kubectl create -f test/fixtures/pkg/kubectl/cmd/create/sar-v1beta1.json --validate=false

  SAR_RESULT_FILE="${KUBE_TEMP}/sar-result.json"
  curl -k -H "Content-Type:" http://localhost:8080/apis/authorization.k8s.io/v1beta1/subjectaccessreviews -XPOST -d @test/fixtures/pkg/kubectl/cmd/create/sar-v1beta1.json > "${SAR_RESULT_FILE}"
  if grep -q '"allowed": true' "${SAR_RESULT_FILE}"; then
    kube::log::status "\"authorization.k8s.io/subjectaccessreviews\" returns as expected: $(cat "${SAR_RESULT_FILE}")"
  else
    kube::log::status "\"authorization.k8s.io/subjectaccessreviews\" does not return as expected: $(cat "${SAR_RESULT_FILE}")"
    exit 1
  fi
  rm "${SAR_RESULT_FILE}"

  SAR_RESULT_FILE="${KUBE_TEMP}/sar-result.json"
  curl -k -H "Content-Type:" http://localhost:8080/apis/authorization.k8s.io/v1/subjectaccessreviews -XPOST -d @test/fixtures/pkg/kubectl/cmd/create/sar-v1.json > "${SAR_RESULT_FILE}"
  if grep -q '"allowed": true' "${SAR_RESULT_FILE}"; then
    kube::log::status "\"authorization.k8s.io/subjectaccessreviews\" returns as expected: $(cat "${SAR_RESULT_FILE}")"
  else
    kube::log::status "\"authorization.k8s.io/subjectaccessreviews\" does not return as expected: $(cat "${SAR_RESULT_FILE}")"
    exit 1
  fi
  rm "${SAR_RESULT_FILE}"

  set +o nounset
  set +o errexit
}

run_impersonation_tests() {
  set -o nounset
  set -o errexit

  kube::log::status "Testing impersonation"

  output_message=$(! kubectl get pods "${kube_flags_with_token[@]:?}" --as-group=foo 2>&1)
  kube::test::if_has_string "${output_message}" 'without impersonating a user'

  if kube::test::if_supports_resource "${csr:?}" ; then
    # --as
    kubectl create -f hack/testdata/csr.yml "${kube_flags_with_token[@]:?}" --as=user1
    kube::test::get_object_assert 'csr/foo' '{{.spec.username}}' 'user1'
    kube::test::get_object_assert 'csr/foo' '{{range .spec.groups}}{{.}}{{end}}' 'system:authenticated'
    kubectl delete -f hack/testdata/csr.yml "${kube_flags_with_token[@]:?}"

    # --as-group
    kubectl create -f hack/testdata/csr.yml "${kube_flags_with_token[@]:?}" --as=user1 --as-group=group2 --as-group=group1 --as-group=,,,chameleon
    kube::test::get_object_assert 'csr/foo' '{{len .spec.groups}}' '4'
    kube::test::get_object_assert 'csr/foo' '{{range .spec.groups}}{{.}} {{end}}' 'group2 group1 ,,,chameleon system:authenticated '
    kubectl delete -f hack/testdata/csr.yml "${kube_flags_with_token[@]:?}"
  fi

  set +o nounset
  set +o errexit
}
