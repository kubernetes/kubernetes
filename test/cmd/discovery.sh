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

run_RESTMapper_evaluation_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing RESTMapper"

  RESTMAPPER_ERROR_FILE="${KUBE_TEMP}/restmapper-error"

  ### Non-existent resource type should give a recognizeable error
  # Pre-condition: None
  # Command
  kubectl get "${kube_flags[@]:?}" unknownresourcetype 2>"${RESTMAPPER_ERROR_FILE}" || true
  if grep -q "the server doesn't have a resource type" "${RESTMAPPER_ERROR_FILE}"; then
    kube::log::status "\"kubectl get unknownresourcetype\" returns error as expected: $(cat "${RESTMAPPER_ERROR_FILE}")"
  else
    kube::log::status "\"kubectl get unknownresourcetype\" returns unexpected error or non-error: $(cat "${RESTMAPPER_ERROR_FILE}")"
    exit 1
  fi
  rm "${RESTMAPPER_ERROR_FILE}"
  # Post-condition: None

  set +o nounset
  set +o errexit
}

run_assert_short_name_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing assert short name"

  kube::log::status "Testing propagation of short names for resources"
  output_message=$(kubectl get --raw=/api/v1)

  ## test if a short name is exported during discovery
  kube::test::if_has_string "${output_message}" '{"name":"configmaps","singularName":"","namespaced":true,"kind":"ConfigMap","verbs":\["create","delete","deletecollection","get","list","patch","update","watch"\],"shortNames":\["cm"\],"storageVersionHash":'

  set +o nounset
  set +o errexit
}

run_assert_categories_tests() {
  set -o nounset
  set -o errexit

  kube::log::status "Testing propagation of categories for resources"
  output_message=$(kubectl get --raw=/api/v1 | grep -o '"name":"pods"[^}]*}')
  kube::test::if_has_string "${output_message}" '"categories":\["all"\]'

  set +o nounset
  set +o errexit
}

run_resource_aliasing_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing resource aliasing"
  kubectl create -f test/e2e/testing-manifests/statefulset/cassandra/controller.yaml "${kube_flags[@]}"
  kubectl create -f test/e2e/testing-manifests/statefulset/cassandra/service.yaml "${kube_flags[@]}"

  object="all -l'app=cassandra'"
  request="{{range.items}}{{range .metadata.labels}}{{.}}:{{end}}{{end}}"

  # all 4 cassandra's might not be in the request immediately...
  # :? suffix is for possible service.kubernetes.io/headless
  # label with "" value
  kube::test::get_object_assert "$object" "$request" '(cassandra:){2}(cassandra:(cassandra::?)?)?'

  kubectl delete all -l app=cassandra "${kube_flags[@]}"

  set +o nounset
  set +o errexit
}

run_kubectl_explain_tests() {
  set -o nounset
  set -o errexit

  kube::log::status "Testing kubectl(v1:explain)"
  kubectl explain pods
  # shortcuts work
  kubectl explain po
  kubectl explain po.status.message
  # cronjob work
  kubectl explain cronjob

  set +o nounset
  set +o errexit
}

run_swagger_tests() {
  set -o nounset
  set -o errexit

  kube::log::status "Testing swagger"

  # Verify schema
  file="${KUBE_TEMP}/schema.json"
  curl -kfsS -H 'Authorization: Bearer admin-token' "https://127.0.0.1:${SECURE_API_PORT}/openapi/v2" > "${file}"
  grep -q "list of returned" "${file}"
  grep -q "List of services" "${file}"
  grep -q "Watch for changes to the described resources" "${file}"

  set +o nounset
  set +o errexit
}
